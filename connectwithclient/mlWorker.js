// server/ml_computer/mlWorker.js
require('dotenv').config(); // AWS 자격증명 등 .env 파일
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const { S3Client } = require('@aws-sdk/client-s3');
const { Upload } = require('@aws-sdk/lib-storage');
const { db } = require('./firebaseAdmin');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const { URL } = require('url');

// 현재 디렉토리 내 inputs 및 outputs 폴더 경로 설정
const currentDir = __dirname;
const inputsDir = path.join(currentDir, 'inputs');
const outputsDir = path.join(currentDir, 'outputs');

// 폴더가 존재하지 않으면 생성
if (!fs.existsSync(inputsDir)) {
  fs.mkdirSync(inputsDir, { recursive: true });
  console.log(`Created inputs directory at ${inputsDir}`);
}

if (!fs.existsSync(outputsDir)) {
  fs.mkdirSync(outputsDir, { recursive: true });
  console.log(`Created outputs directory at ${outputsDir}`);
}

async function main() {
  console.log('ML Worker is starting...');

  // Firestore collectionGroup 실시간 리스너
  db.collectionGroup('shots')
    .where('analysis', '==', null)
    .onSnapshot(
      (snapshot) => {
        snapshot.docChanges().forEach(async (change) => {
          if (change.type === 'added') {
            const doc = change.doc;
            const docData = doc.data();

            console.log('New unprocessed doc found:', doc.id, docData);

            // 1) S3 URL에서 영상 다운로드
            const videoUrl = docData.s3Url;
            const hand = docData.hand; // 추가: handedness 정보
            if (!videoUrl) {
              console.log('No s3Url found. Skip.');
              return;
            }

            // 2) inputs 폴더에 파일 저장
            const inputVideoPath = path.join(inputsDir, `input_${doc.id}${path.extname(videoUrl)}`);
            const outputVideoPath = path.join(outputsDir, `output_processed_${doc.id}.mp4`);

            try {
              await downloadFile(videoUrl, inputVideoPath);
              console.log('Video downloaded to', inputVideoPath);

              // 3) 머신러닝 분석 수행 (hand 정보 전달)
              const { overall_similarity, output_video_path, analysis } = await runMLProcess(inputVideoPath, outputVideoPath, hand); // 수정: 'hand' 전달
              console.log('ML Analysis done:', { overall_similarity, output_video_path, analysis });

              // 4) 새 영상 S3 업로드
              const newS3Key = `processed_${doc.id}.mp4`;
              const newS3Url = await uploadFileToS3(output_video_path, newS3Key);
              console.log('Uploaded new processed video to', newS3Url);

              // 5) Firestore 문서 업데이트
              await doc.ref.update({
                analysis,
                score: overall_similarity,
                newUrl: newS3Url,
                processed: new Date(), // 처리 완료 시간
              });

              console.log(`Document ${doc.id} updated with ML results.`);
            } catch (err) {
              console.error('Error in ML pipeline:', err);

              // Firestore 문서 업데이트: '검출 실패' 설정
              try {
                await doc.ref.update({
                  analysis: '검출 실패',
                  score: 0, // 필요한 경우 스코어도 초기화
                  processed: new Date(), // 처리 완료 시간
                });
                console.log(`Document ${doc.id} updated with '검출 실패'.`);
              } catch (updateErr) {
                console.error(`Failed to update document ${doc.id} with '검출 실패':`, updateErr);
              }
            }
          }
        });
      },
      (error) => {
        console.error('onSnapshot error:', error);
      }
    );

  console.log('ML Worker is now listening for new unprocessed shots...');
}

async function runMLProcess(inputVideoPath, outputVideoPath, hand) { // 수정: 'hand' 인자 추가
  return new Promise((resolve, reject) => {
    console.log('Starting shot_analyzer.py with input:', inputVideoPath, 'and output:', outputVideoPath, 'hand:', hand);

    // Python 스크립트 경로
    const pythonScriptPath = path.join(__dirname, 'shot_analyzer.py');

    // Python 프로세스 실행: handedness 인자 추가
    const pythonProcess = spawn('python3', [pythonScriptPath, inputVideoPath, outputVideoPath, hand]); // 수정: 'hand' 전달

    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`shot_analyzer.py exited with code ${code}`);
        console.error('stderr:', stderrData);
        return reject(new Error(`shot_analyzer.py failed with code ${code}`));
      }

      try {
        // JSON 데이터만 추출
        const jsonStart = stdoutData.indexOf('{');
        if (jsonStart === -1) {
          throw new Error('No JSON output found in shot_analyzer.py output');
        }
        const jsonString = stdoutData.slice(jsonStart);
        const result = JSON.parse(jsonString);

        if (!result.success) {
          console.error('shot_analyzer.py reported failure:', result.message);
          return reject(new Error(result.message));
        }

        const overall_similarity = result.overall_similarity;
        const analysis = result;
        const output_video_path = result.output_video_path; // 이미 절대 경로임

        resolve({ overall_similarity, output_video_path, analysis });
      } catch (err) {
        console.error('Failed to parse JSON from shot_analyzer.py:', err);
        console.error('stdout:', stdoutData);
        return reject(new Error('Invalid JSON output from shot_analyzer.py'));
      }
    });
  });
}

async function downloadFile(url, destPath) {
  // URL 인코딩
  let encodedUrl;
  try {
    encodedUrl = new URL(url).toString();
  } catch (err) {
    console.error(`Invalid URL: ${url}`);
    throw new Error(`Invalid URL: ${url}`);
  }

  try {
    const response = await axios({
      method: 'GET',
      url: encodedUrl,
      responseType: 'stream',
      validateStatus: function (status) {
        return status >= 200 && status < 300; // 2xx 상태 코드만 성공으로 간주
      },
    });

    // Content-Type 확인
    const contentType = response.headers['content-type'];
    console.log(`Downloaded content type: ${contentType}`);
    if (!contentType.startsWith('video/')) {
      throw new Error(`Invalid content type: ${contentType}`);
    }

    return new Promise((resolve, reject) => {
      const writer = fs.createWriteStream(destPath);
      response.data.pipe(writer);
      let error = null;
      writer.on('error', (err) => {
        error = err;
        writer.close();
        reject(err);
      });
      writer.on('close', () => {
        if (!error) {
          // 파일 크기 확인 (예: 1KB 이상)
          fs.stat(destPath, (err, stats) => {
            if (err) {
              return reject(err);
            }
            console.log(`Downloaded file size: ${stats.size} bytes`);
            if (stats.size < 1024) {
              return reject(new Error('Downloaded file is too small, possibly corrupted.'));
            }
            resolve();
          });
        }
      });
    });
  } catch (error) {
    console.error(`Failed to download file from ${url}:`, error.message);
    throw new Error(`Failed to download file from ${url}`);
  }
}

/**
 * S3 업로드: @aws-sdk/lib-storage 사용
 */
async function uploadFileToS3(filePath, s3Key) {
  const s3 = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
    },
  });
  const fileStream = fs.createReadStream(filePath);

  const upload = new Upload({
    client: s3,
    params: {
      Bucket: process.env.S3_BUCKET_NAME,
      Key: s3Key,
      Body: fileStream,
      ContentType: 'video/mp4',
    },
  });

  await upload.done();

  return `https://${process.env.S3_BUCKET_NAME}.s3.amazonaws.com/${s3Key}`;
}

// 실행
main().catch((err) => {
  console.error('ML Worker crashed:', err);
});