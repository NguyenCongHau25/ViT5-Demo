# VietNorm

Dự án dùng ngôn ngữ T5 (Seq2Seq) để huấn luyện dịch và khôi phục ngôn ngữ tiếng Việt (dịch phương ngữ, sửa lỗi không dấu, sai chính tả).

## Cấu trúc
* **backend**: REST API dùng FastAPI và mô hình HuggingFace Transformers.
* **frontend**: Ứng dụng giao diện web với Next.js và Tailwind CSS. 

## Cài đặt nhanh

Khởi chạy cả Frontend và Backend bằng Docker Compose:
`bash
docker-compose up --build
`

Hoặc chạy độc lập: 
* Backend: `cd backend && uvicorn main:app --reload --port 7860`
* Frontend: `cd frontend && npm install && npm run dev`
