const express = require('express');
const cors = require('cors');
const app = express();
const axios = require('axios');
const net = require('net');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

const HTTP_PORT = 4002;  // Express 서버 포트
const TCP_PORT = 4001;   // TCP 서버 포트

const server_ip = '192.168.0.8'; // 서버 IP
let recognizedProducts = [];  // 인식된 상품을 추적하는 배열

let currentPage = null;  // 현재 페이지 정보 전역 변수

let tcpServer = null; // TCP 서버 인스턴스

app.use(express.json());

const corsOptions = {
  origin: '*',  // 모든 도메인 허용 (보안 설정이 필요할 경우, 특정 도메인만 허용)
  methods: ['GET', 'POST'],  // 허용할 HTTP 메서드 지정
  allowedHeaders: ['Content-Type', 'Authorization'],  // 허용할 헤더 지정
};
app.use(cors(corsOptions));


// 페이지 정보 수신 및 업데이트
app.post('/page-info', async (req, res) => {
  const { currentPage: page } = req.body;

  // 유효하지 않은 페이지 필터링
  if (![0, 2, 4].includes(page)) {
    console.log(`유효하지 않은 페이지: ${page}`);
    res.status(400).send("유효하지 않은 페이지 정보입니다.");
    return;
  }

  console.log(`수신된 페이지 정보: ${page}`);
  
 
  // 페이지가 변경되었을 때만 handlePageInfo 실행
  if (page !== currentPage || page === 4) {
    try {
      console.log(`페이지 ${currentPage}에서 ${page}로 전환됨, 처리 시작.`);
      await handlePageInfo(page);  
      currentPage = page;
      console.log(`페이지 ${page}로 전환됨. 처리 완료.`);
      res.send(`페이지 ${page} - 처리 준비 완료\n`);
    } catch (error) {
      console.error("페이지 처리 중 오류 발생:", error);
      res.status(500).send("페이지 처리 중 오류 발생");
    }
  } else {
    console.log(`이미 ${page} 페이지 상태 유지 중.`);
    res.send(`이미 ${page} 페이지 상태 유지 중\n`);
  }
});



async function handlePageInfo(page) {
  try {
    if (page === 4) {
      console.log("페이지 4 - 카메라 미리 렌더링");
      await startTCPServer(page);
      await startPythonServer(page);
      // yolo off

    } else if (page === 0) {
      console.log("0번 페이지 - YOLO 활성화");
       await startTCPServer(page);  // TCP 서버 시작
       await startPythonServer(page); // 카메라 렌더링 및 Python 서버 실행
       await setYoloActive(true);   // YOLO 활성화
       

    } else if (page === 2) {
      console.log("2번 페이지 - 상품 인식 YOLO 활성화");
      await startTCPServer(page);    // 2페이지용 TCP 생성
      console.log("TCP 서버 생성 완료, Python 서버 시작 시도");
      await startPythonServer(page); // 카메라 렌더링 및 Python 서버 실행
      console.log("Python 서버 실행 요청 완료");
      await setYoloActive(true);     // YOLO 활성화
    }
  } catch (error) {
    console.error(`페이지 ${page} 처리 중 오류 발생:`, error);
  } finally {
      isProcessing = false;
  }
}


async function setYoloActive(isActive) {
  try {
    await fetch("http://192.168.0.8:5003/toggle_yolo", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ yolo_active: isActive })
    });
    console.log(`YOLO 활성화 상태가 ${isActive ? 'ON' : 'OFF'}로 설정되었습니다.`);
  } catch (error) {
    console.error("YOLO 활성화 상태 설정 오류:", error);
  }
}


// Python 서버 시작 함수
async function startPythonServer(page) {
  try {
    const response = await axios.get("http://192.168.0.8:5003/status");
    console.log("Python 서버 상태 확인 응답:", response.status); // 추가
    if (response.status === 200) {
      console.log("Python 서버가 이미 실행 중입니다. 페이지 상태 업데이트 요청 중...");
      
      // Python 서버의 페이지 상태를 업데이트
      await axios.post("http://192.168.0.8:5003/update_page", { page });
      return;
    }
  } catch (error) {
    console.log("Python 서버가 실행되지 않음. 서버를 시작합니다.");
    try {
      await execAsync(`python D:/robot_kiosk/deep-text-recognition-benchmark/OCR_test.py ${page}`);
      console.log("Python 서버가 성공적으로 시작되었습니다.");
    } catch (err) {
      console.error("Python 서버 시작 중 오류 발생:", err.message);
    }
  }
}



// YOLO 데이터 처리 함수
async function handleYOLOData(data, socket, page) {
  const classId = Buffer.isBuffer(data) ? data.toString('utf8').trim() : data.trim();
  console.log(`YOLO로부터 처리할 데이터: ${classId}, 현재 페이지: ${page}`);

   
    // END 데이터 처리
    if (classId === 'END') {
      console.log('YOLO 데이터가 END입니다. 상품 인식 중단');
      if (!socket.destroyed) {
        socket.end();
        socket.destroy(); // 소켓 강제 종료
      }
      return; // END 신호 처리 완료
    }
  
  if (page === 4) {
    return;
  }

  try {
    if (page === 0) {
      // 0페이지 - text 데이터 처리
      const recognizedText = classId; // OCR 텍스트 데이터로 처리
   
      const isCustomerRecognized = await sendTextToRecognitionServer(recognizedText);
      if (isCustomerRecognized) {
        if (!socket.destroyed) {
          socket.write('종료', () => {
            socket.end();
            socket.destroy(); // 소켓을 강제로 종료
          });
        }
        socket.removeAllListeners('data');
      }
      else {
        if (!socket.destroyed) socket.write('대기');
      }

    } else if (page === 2) {
      const isProductRecognized = await sendProductToRecognitionServer(classId);

      if (!recognizedProducts.includes(classId) && isProductRecognized) {
        recognizedProducts.push(classId);
        console.log(`상품 인식: ${classId}, 총 인식된 상품 수: ${recognizedProducts.length}`);
      
        if (recognizedProducts.length >= 4) {
          recognizedProducts = []; // 상품 리스트 초기화
          if (!socket.destroyed) {
            socket.write('END', () => {
              socket.end();
              socket.destroy(); // 소켓 종료
            });
          }
        }
      }
    }
  } catch (error) {
    console.error('데이터 처리 중 오류 발생:', error);
  }
}

//---------------- 텍스트 데이터 -> 닉네임 인식 서버로 전송 -------------------------------
async function sendTextToRecognitionServer(text) {
  const url = 'http://192.168.0.8:5001/sticker-recognition';  // 0페이지 URL
  console.log('보내기 전 text 값:', text);

  try {
    const response = await axios.post(url, { text }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });

    console.log(`닉네임 인식 서버 응답: ${response.status} ${response.statusText}`);
    if (response.status === 200 && response.data.status === 'success') {
      return true;  // 인식 성공
    } else {
      return false;  // 인식 실패
    }
  } catch (error) {
    const statusCode = error.response ? error.response.status : null;

    if (statusCode === 404) {
      console.error(`닉네임 인식 서버 오류: 404 - URL을 찾을 수 없습니다: ${url}`);
      return false;
    }

    console.error(`닉네임 인식 서버 오류: ${error.message || statusCode}`);
    return false;
  }
}

//---------------- 상품 데이터 -> 상품 인식 서버로 전송 -------------------------------
async function sendProductToRecognitionServer(classId) {
  const url = 'http://192.168.0.8:5001/product-recognition';  // 2페이지 URL
  console.log(`2페이지 데이터 전송: ${classId}`);

  try {
    const response = await axios.post(url, { class_id: classId }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });

    console.log(`상품 인식 서버 응답: ${response.status} ${response.statusText}`);
    if (response.status === 200 && response.data.status === 'success') {
      return true;  // 인식 성공
    } else {
      return false;  // 인식 실패
    }
  } catch (error) {
    const statusCode = error.response ? error.response.status : null;

    if (statusCode === 404) {
      console.error(`상품 인식 서버 오류: 404 - URL을 찾을 수 없습니다: ${url}`);
      return false;
    }

    console.error(`상품 인식 서버 오류: ${error.message || statusCode}`);
    return false;
  }
}


async function startTCPServer(page) {
  console.log(`TCP 서버를 페이지 ${page}용으로 시작합니다.`);

  // 기존 서버 종료
  if (tcpServer) {
    console.log('기존 TCP 서버 종료 시도 중...');
    await new Promise((resolve) => {
      tcpServer.close(() => {
        console.log('기존 TCP 서버가 안전하게 종료되었습니다.');
        resolve();
      });
    });
    tcpServer = null; // 종료 후 초기화
  }

  // 새로운 서버 생성
  tcpServer = net.createServer((socket) => {
    console.log(`TCP 클라이언트가 페이지 ${page}에 연결되었습니다.`);

    socket.on('data', async (data) => {
      console.log(`수신된 데이터: ${data}, 현재 페이지: ${page}`);
      if (currentPage === page) {
        await handleYOLOData(data, socket, page);
      }
    });

    socket.on('error', (err) => console.error(`소켓 에러: ${err.message}`));
    socket.on('end', () => console.log('클라이언트와 연결이 종료되었습니다.'));
  });

  // 서버 에러 처리
  tcpServer.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.error('포트가 이미 사용 중입니다. 기존 프로세스를 확인하세요.');
    } else {
      console.error(`TCP 서버 에러: ${err.message}`);
    }
  });

  // 서버 시작
  await new Promise((resolve) => {
    tcpServer.listen(TCP_PORT, server_ip, () => {
      console.log(`TCP 서버가 ${TCP_PORT} 포트에서 페이지 ${page}용으로 대기 중입니다.`);
      resolve();
    });
  });
}

// Express 서버 실행 (HTTP 요청 처리)
app.listen(HTTP_PORT, () => {
  console.log(`Express 서버가 ${HTTP_PORT} 포트에서 실행 중입니다.`);
});
