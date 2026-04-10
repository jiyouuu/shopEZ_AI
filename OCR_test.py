import string
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from aiortc.rtcicetransport import RTCIceCandidate

from ocr_utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import os
from torchvision import transforms

from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import socket
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
from av import VideoFrame
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import asyncio
import sys
from starlette.requests import Request

# YOLO utils 디렉토리 설정
yolov5_dir = os.path.join(os.path.dirname(__file__), "yolov5")
sys.path.insert(0, yolov5_dir)

# 로깅 설정
logging.basicConfig(filename='server_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# 페이지 정보를 받음
page = int(sys.argv[1]) if len(sys.argv) > 1 else None

# TCP 클라이언트 소켓 전역 변수
client = None
is_connected = False  # 연결 상태를 추적
# 전역 변수
detected_text = None  # 초기값 설정

# 서버 연결 정보
server_ip = '192.168.0.8'  # 서버 IP
server_port = 4001  # 서버 포트


# RTCPeerConnection 인스턴스를 전역으로 관리
peer_connections = []


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 전역 변수
ocr_model = None
ocr_converter = None
preprocess = None
tensor_transform = None
opt = None

# OCR 모델 초기화 함수
def initialize_ocr():
    global ocr_model, ocr_converter, preprocess, tensor_transform, opt

    # 옵션 설정
    opt = Opt()

    if 'CTC' in opt.Prediction:
        ocr_converter = CTCLabelConverter(opt.character)
    else:
        ocr_converter = AttnLabelConverter(opt.character)
    opt.num_class = len(ocr_converter.character)
    opt.input_channel = 1 # 흑백 이미지 사용

    # 모델 설정 및 로드
    ocr_model = Model(opt)
    ocr_model = torch.nn.DataParallel(ocr_model).to(device)

    # 사전 학습된 모델 로드
    ocr_model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    ocr_model.eval()  # 평가 모드

    # 이미지 전처리 파이프라인 정의 (Normalize 제외)
    preprocess = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgW), interpolation=Image.BICUBIC)
    ])
    
    # 텐서 변환 및 정규화 파이프라인 정의
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


# RTCPeerConnection을 생성할 때마다 추가
def create_peer_connection():
    pc = RTCPeerConnection()
    peer_connections.append(pc)

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(f"ICE Connection State: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            peer_connections.remove(pc)
    
    return pc

def get_peer_connection(client_id):
    return peer_connections.get(client_id)

# 모든 PeerConnection을 안전하게 닫는 함수
async def close_peer_connections():
    logging.info("모든 PeerConnection을 닫습니다.")
    close_tasks = [pc.close() for pc in peer_connections]
    await asyncio.gather(*close_tasks)
    logging.info("모든 PeerConnection이 닫혔습니다.")


# 서버 소켓 생성 및 연결 함수 (4번 페이지일 경우 연결하지 않음)
async def connect_to_server():
    global client, is_connected
    if page == 4:
        logging.info("4번 페이지 - TCP 연결 생략")
        return

 # 연결 시도를 한 번만 실행하여 중복 연결 방지
    if not is_connected:
        while not is_connected:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect((server_ip, server_port))
                is_connected = True
                logging.info("서버에 성공적으로 연결되었습니다.")
            except Exception as e:
                logging.error(f"서버 연결 오류: {e}. 5초 후 재시도합니다.")
                await asyncio.sleep(5)



# YOLO 활성화 상태 및 전역 변수
yolo_active = False

model = None
last_detections = None
last_prediction = None
same_pred_count = 0

# detection_lock = threading.Lock()
detection_lock = asyncio.Lock()  # YOLO 결과 보호를 위한 비동기 락
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 클래스 ID별 감지 횟수 저장
class_id_count = {}

# 페이지별 고유 클래스 추적용 변수
recognized_classes_2 = []


# YOLO 모델 로드 함수
def load_yolo_model(page):
    global model
    if page == 2:
        # 2페이지: 상품 인식 모델 로드
        model_path = 'D:/robot_kiosk/deep-text-recognition-benchmark/best.pt'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(device)
        model.conf = 0.5  # confidence threshold
        model.iou = 0.4  # IoU threshold
        logging.info("2페이지: 상품 인식 모델 로드 완료")
        logging.info(f"YOLO 모델 이름: {model.names}")
    else:
        logging.warning(f"{page} 페이지에 대한 모델이 없습니다.")
        return None
    return model

# YOLO 활성화 상태 설정
def set_yolo_active(active):
    global yolo_active, isCameraInitialized,is_connected,model
    yolo_active = active
    logging.info(f"YOLO 활성화 상태: {yolo_active}")

    if not yolo_active:
        logging.info("YOLO 비활성화 상태로 전환됩니다.")
        yolo_active = False
        model = None

        if client:
            try:
                client.close()
                is_connected = False
                client = None
                logging.info("TCP 클라이언트 연결이 성공적으로 해제되었습니다.")
            except Exception as e:
                logging.error(f"TCP 클라이언트 해제 중 오류 발생: {e}")

    elif page == 2 and yolo_active:  # 페이지가 2일 때만 YOLO 모델 로드
        try:
            logging.info(f"현재 페이지: {page}에 맞는 YOLO 모델을 로드합니다.")
            # 현재 페이지에 맞는 YOLO 모델 로드
            load_yolo_model(page)
            logging.info(f"YOLO 모델이 페이지 {page}에 맞게 로드되었습니다.")
        except Exception as e:
            logging.error(f"YOLO 모델 로드 중 오류 발생: {e}")
            return

        # YOLO 활성화 시 TCP 서버 연결
        if not is_connected:
            asyncio.create_task(connect_to_server())  # 비동기로 TCP 연결 시도
    else:
        logging.info("YOLO 비활성화 상태로 전환되었습니다.")



# 객체 인식 서버로부터 메시지를 수신하는 함수
async def receive_server_message():
    try:
        message = client.recv(1024).decode('utf-8')
        if message == '종료':
            logging.info("카메라 종료 신호 수신, 카메라를 종료합니다.")
            return True
        elif message == '대기':
            logging.info("대기 신호 수신, 카메라 계속 작동 중...")
            return False
    except Exception as e:
        logging.error(f"서버로부터 메시지 수신 중 오류 발생: {e}")
    return False


async def close_tcp_server():
    global client, is_connected

    # 클라이언트 소켓이 연결된 상태라면 닫음
    if client and is_connected:
        try:
            logging.info("TCP 클라이언트 연결 종료 시도 중...")
            client.shutdown(socket.SHUT_RDWR)  # 읽기/쓰기 종료
            client.close()  # 소켓 닫기
            is_connected = False
            logging.info("TCP 클라이언트 연결이 성공적으로 종료되었습니다.")
        except Exception as e:
            logging.error(f"TCP 클라이언트 연결 종료 중 오류 발생: {e}")
        finally:
            client = None  # 클라이언트 소켓 초기화



# YOLO 연산 함수 (비동기)
async def run_yolo(frame,page):
    global last_detections, ocr_model, ocr_converter, preprocess, tensor_transform, last_prediction, same_pred_count  # 전역 변수 참조
    if page == 4 or not yolo_active:
        return  # 페이지가 4이거나 yolo_active가 False일 경우 YOLO 처리를 중지
    
    if page == 0:
        # 0페이지: 글자체 인식
        logging.info("글자체 인식을 실행합니다.")
        try:
            # 이미지를 흑백으로 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- 노이즈 제거 및 이진화 처리 추가 ---
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            _, binary_frame = cv2.threshold(blurred_frame, 127, 255, cv2.THRESH_BINARY)

            # OpenCV 이미지를 PIL 이미지로 변환
            pil_image = Image.fromarray(binary_frame).convert('L')
            pil_image.save("debug_input_image.png")

            # 밝기 조정
            enhancer_brightness = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer_brightness.enhance(opt.brightness)

            # 대비 조정
            enhancer_contrast = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer_contrast.enhance(opt.contrast)

            # 이미지 전처리 (크기 조정)
            resized_image = preprocess(pil_image)
          
            # 텐서 변환 및 정규화
            image = tensor_transform(resized_image).unsqueeze(0).to(device)
   
            # 예측 수행
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

            with torch.no_grad():
                # 모델 설정
                if 'CTC' in opt.Prediction:
                    preds = ocr_model(image, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)])
                    _, preds_index = preds.max(2)
                    preds_str = ocr_converter.decode(preds_index, preds_size)

                else:
                    preds = ocr_model(image, text_for_pred, is_train=False)
                    _, preds_index = preds.max(2)
                    preds_str = ocr_converter.decode(preds_index, length_for_pred)

                current_prediction = preds_str[0].strip()  # 공백 제거


                 # 빈 문자열 처리
                if not current_prediction:
                    logging.warning("모델 예측 결과가 빈 문자열입니다.")
                    detected_text = None
                else:
                    async with detection_lock:
                        if last_prediction is not None and current_prediction == last_prediction:
                            same_pred_count += 1
                            logging.info(f"동일한 예측 결과 유지: {current_prediction} (동일 카운트: {same_pred_count})")
                        else:
                            same_pred_count = 1  # 새로운 예측값이 들어오면 카운트 초기화
                            logging.info(f"새로운 예측 결과: {current_prediction}")
                        
                          # 마지막 예측 값 업데이트
                        last_prediction = current_prediction
                        # 임계값 도달 시 처리
                        if same_pred_count >= 5:
                            detected_text = current_prediction
                            last_detections = {"text": detected_text}
                            logging.info(f"글자체 인식 결과 저장: {detected_text}")
                            same_pred_count = 0  # 초기화
                        else:
                            last_detections = None  # 임계값 미달


        except Exception as e:
            logging.error(f"글자체 인식 중 오류 발생: {e}")
            async with detection_lock:
                last_detections = None
        return
    
    elif page == 2:
        logging.info("들어옴?")
    # 2페이지: 상품 인식 처리 
        try:
            # YOLOv5 모델 추론
            results = model(frame)

            detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

            if detections.size == 0:
                logging.warning("YOLO 추론 결과가 비어 있습니다.")
                async with detection_lock:
                    last_detections = None
            else:
                boxes, confidences, class_ids, classes = [], [], [], []
                for x1, y1, x2, y2, conf, cls in detections:
                    if conf > 0.5:  # 신뢰도 필터링
                        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # 박스 좌표 저장
                        confidences.append(float(conf))  # 신뢰도 저장
                        class_ids.append(int(cls))  # 클래스 ID 저장
                        classes.append(results.names[int(cls)])  # 클래스 이름 저장
    
             
                # `last_detections`에 저장
                async with detection_lock:
                        last_detections = {
                                "boxes": boxes,
                                "confidences": confidences,
                                "class_ids": class_ids,
                                "classes": classes,
                            }
                        logging.info(f"YOLO 검출 결과 저장: {last_detections}")
                 
        except Exception as e:
            logging.error(f"YOLO 처리 중 오류 발생: {e}")
            async with detection_lock:
                last_detections = None

## 2페이지 상품 인식
async def process_page_2(boxes, confidences, class_ids):
    global recognized_classes_2, class_id_count, client, is_connected

    if page != 2:
        logging.warning(f"process_page_2가 잘못된 페이지에서 호출됨: {page}")
        return

    for class_id in class_ids:
        if class_id not in class_id_count:
            class_id_count[class_id] = 0
        class_id_count[class_id] += 1
        logging.info(f"클래스 ID {class_id} 감지 횟수: {class_id_count[class_id]}")

        if class_id_count[class_id] >= 10:
            try:
                if client and is_connected:
                    # 중복 허용 리스트에 추가 (단순 추가)
                    recognized_classes_2.append(class_id)  
                    client.sendall(f"{class_id}\n".encode())  # 클래스 ID 전송
                    logging.info(f"클래스 ID {class_id}가 서버로 전송되었습니다.")

                # 감지 횟수 초기화
                class_id_count[class_id] = 0

                # 고유 클래스 4개 이상 감지 시 처리
                if len(recognized_classes_2) >= 4:
                    logging.info("4개의 고유 클래스가 감지되었습니다.")
                    recognized_classes_2.clear()  # 감지된 클래스 리스트 초기화
                    client.sendall(b'END\n')  # 서버에 종료 신호 전송
                    logging.info("END 신호가 서버로 전송되었습니다.")
                    await close_tcp_server()  # TCP 서버 종료
                    return  # 함수 종료
                else:
                    logging.info("4개 미만의 상품이 감지됨. 계속 진행합니다.")

            except OSError as e:
                logging.error(f"데이터 전송 중 오류 발생: {e}. 재연결 시도.")
                await connect_to_server()
            except Exception as e:
                logging.error(f"서버 전송 오류: {e}")



# 0페이지 글자체 인식
async def process_page_0(text):
    global detected_text

    if page != 0:
        logging.warning(f"process_page_0가 잘못된 페이지에서 호출됨: {page}")
        return

     # 마지막으로 전송한 텍스트와 비교하여 중복 방지
    if detected_text is not None and text.strip() == detected_text:
        logging.info(f"중복된 텍스트 '{text}'는 전송되지 않습니다.")
        return  # 중복된 텍스트는 전송하지 않음

    try:
        # 서버로 텍스트 전송
        client.sendall(f"{text}\n".encode())
        logging.info(f"텍스트 '{text}'가 서버로 전송되었습니다.")

        # 서버 메시지 대기
        should_exit = await receive_server_message()
        if should_exit:
            logging.info("서버 종료 신호를 수신하여 작업을 종료합니다.")
            return  # 함수 종료
        else:
            logging.info("서버의 '대기' 신호로 카메라를 계속 작동합니다.")

        # **서버 전송 후에 `detected_text` 업데이트**
        detected_text = text.strip()

    except OSError as e:
        logging.error(f"데이터 전송 중 오류 발생: {e}. 재연결 시도.")
        await connect_to_server()
    except Exception as e:
        logging.error(f"서버 전송 오류: {e}")



# WebRTC 비디오 스트림 트랙
class OpenCVVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)

        # 카메라 해상도 설정 (1920x1080으로 설정)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 너비
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 높이

        if not self.cap.isOpened():
            logging.error("카메라를 열지 못했습니다.")
            raise Exception("카메라 초기화 실패")
        else:
            logging.info("카메라가 성공적으로 열렸습니다.")
        self.frame_counter = 0  # 프레임 간격 제어를 위한 카운터
        self.frame_skip = 5     # YOLO 실행 간격 (5번째 프레임마다 실행)

    async def recv(self):
        global last_detections  # 전역에서 카메라 객체를 사용
    
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()

        if not ret:
            logging.error("카메라에서 프레임을 가져올 수 없습니다.")
            return VideoFrame.from_ndarray(np.zeros((720, 1280, 3), dtype=np.uint8), format="rgb24")
      
        # 페이지 4인 경우 YOLO 처리 제외
        if page == 4:
            logging.info("현재 페이지는 4입니다. YOLO 처리를 건너뜁니다.")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame
        
        # YOLO 활성화 시 특정 프레임 간격으로 객체 인식 수행
        if yolo_active and self.frame_counter % self.frame_skip == 0:
            asyncio.create_task(run_yolo(frame.copy(), page))  # 비동기로 실행

        self.frame_counter += 1


         # YOLO 검출 결과 화면에 표시
        async with detection_lock:
               if last_detections and isinstance(last_detections, dict):
                  if page == 0 and "text" in last_detections:
                    # 글자체 인식 결과 표시
                    detected_text = last_detections["text"]
                    height, width, _ = frame.shape
                    x_pos, y_pos = width // 4, height // 10
                    cv2.putText(frame, f"Detected Text: {detected_text}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if detected_text:
                        await process_page_0(detected_text)
                    else:
                        logging.warning("글자체 인식 데이터가 유효하지 않음")
              
                  elif page == 2 and "boxes" in last_detections:
                        boxes = last_detections["boxes"]
                        confidences = last_detections["confidences"]
                        classes = last_detections["classes"]
                        class_ids = last_detections["class_ids"]

                        if boxes and confidences and class_ids:
                            logging.info(f"상품 인식 페이지 처리 중, 감지된 클래스: {classes}")

                            # 감지된 객체 표시
                            for i in range(len(boxes)):
                                x, y, w, h = boxes[i]
                                label = f"{classes[i]}: {confidences[i]:.2f}"
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 바운딩 박스
                                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                logging.info(f"상품 {classes[i]} (ID: {class_ids[i]}), 신뢰도: {confidences[i]:.2f}, 박스: {x}, {y}, {w}, {h}")

                            # `process_page_2`를 호출하여 추가 처리 수행
                            await process_page_2(boxes, confidences, class_ids)
        
              
                  elif page == 4:
                    # 페이지 4일 경우 처리 생략
                    logging.info("페이지 4에서는 YOLO 결과 처리를 생략합니다.")
                  else:
                    logging.error(f"알 수 없는 페이지 값: {page}")
                        
     
        # WebRTC로 전송할 비디오 프레임 생성
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
    
    async def stop(self):
        # PeerConnection 닫힐 때 카메라 해제
        if self.cap.isOpened():
            self.cap.release()
        await super().stop()





# Starlette 앱 설정
app = Starlette()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.0.8:3000", "http://localhost:3000", "http://localhost:3001","http://192.168.0.8:3001"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.route("/candidate", methods=["POST"])
async def candidate(request):
    params = await request.json()
    candidate = params.get("candidate")
    sdp_mid = params.get("sdpMid")
    sdp_mline_index = params.get("sdpMLineIndex")

    if not candidate or not sdp_mid or sdp_mline_index is None:
        return JSONResponse({"error": "ICE Candidate 데이터가 누락되었습니다"}, status_code=400)

    # PeerConnection 가져오기
    pc = get_peer_connection()  # 적절한 방식으로 PC를 가져오도록 구현 필요
    ice_candidate = RTCIceCandidate(
        candidate=candidate,
        sdpMid=sdp_mid,
        sdpMLineIndex=sdp_mline_index,
    )
    await pc.addIceCandidate(ice_candidate)

    return JSONResponse({"status": "ICE Candidate 추가 완료"})

# WebRTC 오퍼 요청
@app.route("/offer", methods=["POST"])
async def offer(request):
    params = await request.json()
    sdp = params.get("sdp")
    type_ = params.get("type")

    if not sdp or not type_:
        return JSONResponse({"error": "SDP 또는 Type이 누락되었습니다"}, status_code=400)

    pc = create_peer_connection()
    pc.addTrack(OpenCVVideoStreamTrack())
    
    offer = RTCSessionDescription(sdp=sdp, type=type_)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    response = JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# YOLO 활성화/비활성화 API
@app.route("/toggle_yolo", methods=["POST"])
async def toggle_yolo(request):
    data = await request.json()
    yolo_status = data.get("yolo_active", False)
    set_yolo_active(yolo_status)
    return JSONResponse({"message": f"YOLO 활성화 상태가 {yolo_status}로 설정되었습니다."})


# 서버 상태 확인용 엔드포인트
@app.route("/status", methods=["GET"])
async def status(request):
    response = JSONResponse({
        "status": "Python server is ready",
        "isWebRTCConnected": any(pc.iceConnectionState == "connected" for pc in peer_connections)
    })
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# 페이지 상태를 업데이트하는 API 엔드포인트
@app.route("/update_page", methods=["POST"])
async def update_page(request: Request):
    global page,last_prediction
    data = await request.json()
    new_page = data.get("page", 4)

    # 페이지가 변경되었는지 확인하는 로그 추가
    logging.info(f"요청된 페이지: {new_page}, 현재 페이지: {page}")

    # 페이지 변경 시 기존 연결 종료
    if new_page != page:
        logging.info(f"페이지가 {page}에서 {new_page}로 변경됩니다.")
        await close_tcp_server()  # TCP 서버 종료
        last_prediction = None          # 텍스트 예측 초기화
    else:
        logging.info("페이지가 변경되지 않았습니다.")

    page = new_page  # 페이지 업데이트
    logging.info(f"페이지 상태가 {page}로 업데이트되었습니다.")

    # TCP 연결 필요 시 재연결
    if page == 0 and not is_connected:
        await connect_to_server()

    if page == 2 and not is_connected:
        await connect_to_server()

    if page == 4 and not is_connected:
        await connect_to_server()
    return JSONResponse({"message": f"페이지 상태가 {page}로 업데이트되었습니다."})

app.add_route("/", lambda request: JSONResponse({"status": "WebRTC 서버 실행 중"}))

if __name__ == "__main__":
    import uvicorn
    # asyncio.run(main())  # main()을 실행하여 서버에 연결

    class Opt:
        saved_model = "D:/robot_kiosk//deep-text-recognition-benchmark/best_accuracy.pth"

        batch_max_length = 25
        imgH = 32
        imgW = 100
        rgb = False
        brightness = 2.0
        contrast = 1.5
        korean_chars = ''.join(chr(i) for i in range(ord('가'), ord('힣') + 1))
        english_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        character = korean_chars + english_chars
        Transformation = 'TPS'
        FeatureExtraction = 'ResNet'
        SequenceModeling = 'BiLSTM'
        Prediction = 'CTC'
        num_fiducial = 20
        input_channel = 1
        output_channel = 512
        hidden_size = 256
        PAD = True

    cudnn.benchmark = True
    cudnn.deterministic = True

    initialize_ocr()  # OCR 초기화

    uvicorn.run(app, host="192.168.0.8", port=5003)
