// ml_computer/mlWorker.js
require('dotenv').config(); // AWS 자격증명 등 .env 파일
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const { S3Client } = require('@aws-sdk/client-s3');
const { Upload } = require('@aws-sdk/lib-storage');

const { db } = require('./firebaseAdmin');

async function main() {
  console.log('ML Worker is starting...');

  // Firestore collectionGroup 실시간 리스너
  db.collectionGroup('shots')
    .where('analysis', '==', null)
    .onSnapshot(
      async (snapshot) => {
        snapshot.docChanges().forEach(async (change) => {
          if (change.type === 'added') {
            const doc = change.doc;
            const docData = doc.data();

            console.log('New unprocessed doc found:', doc.id, docData);

            // 1) S3 URL에서 영상 다운로드
            const videoUrl = docData.s3Url;
            if (!videoUrl) {
              console.log('No s3Url found. Skip.');
              return;
            }

            // 2) 임시 폴더에 다운로드
            const tempFilePath = path.join(__dirname, `temp_${doc.id}.mp4`);
            try {
              await downloadFile(videoUrl, tempFilePath);
              console.log('Video downloaded to', tempFilePath);

              // 3) 머신러닝 분석 (여기서는 Mock)
              const { score, analysis, newVideoPath } = await runMLProcess(tempFilePath);
              console.log('ML Analysis done:', { score, analysis, newVideoPath });

              // 4) 새 영상 S3 업로드
              const newS3Url = await uploadFileToS3(newVideoPath, `processed_${doc.id}.mp4`);
              console.log('Uploaded new processed video to', newS3Url);

              // 5) Firestore 문서 업데이트
              await doc.ref.update({
                analysis,
                score,
                newUrl: newS3Url,
                processed: new Date(), // 처리 완료 시간
              });

              console.log(`Document ${doc.id} updated with ML results.`);
            } catch (err) {
              console.error('Error in ML pipeline:', err);
            } finally {
              // 임시 파일 제거
              if (fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
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

async function runMLProcess(inputVideoPath) {
  // 실제 분석 로직은 여기서 수행 (Mock)
  const mockScore = Math.floor(Math.random() * 100);
  const mockAnalysis = 'This is a mock analysis result.';

  // "새 영상"도 그냥 원본 복사
  const newVideoPath = inputVideoPath.replace('.mp4', '_processed.mp4');
  fs.copyFileSync(inputVideoPath, newVideoPath);

  return {
    score: mockScore,
    analysis: mockAnalysis,
    newVideoPath,
  };
}

async function downloadFile(url, destPath) {
  const response = await axios({
    method: 'GET',
    url,
    responseType: 'arraybuffer',
  });
  fs.writeFileSync(destPath, response.data);
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
