"use client";

import { useState } from "react";
import { Sparkles, Languages } from "lucide-react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setResult("Đang dịch...");

    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: prompt.trim(), max_length: 128 }),
      });

      const data = await response.json();
      if (data.error) {
        setResult(`Lỗi: ${data.error}`);
      } else {
        setResult(data.result);
      }
    } catch (e) {
      setResult("Không thể kết nối đến máy chủ AI.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fc] font-sans flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 py-4 px-8 flex items-center gap-4 shadow-sm">
        <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white">
          <Sparkles className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-gray-900 leading-tight">Dịch Phương Ngữ</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-10 flex flex-col relative z-0">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            ViDialect
          </h2>
          <p className="text-md text-gray-600 mt-2 max-w-3xl">
            Nhập văn bản tiếng địa phương vào ô bên dưới, hệ thống của chúng tôi sẽ dịch sang tiếng Việt phổ thông chuẩn mực và dễ hiểu nhất.
          </p>
        </div>

        {/* Translation Container */}
        <div className="relative">
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col md:flex-row min-h-[400px]">
            {/* Input Section */}
            <div className="flex-1 p-6 flex flex-col border-b md:border-b-0 md:border-r border-gray-200 relative pb-16">
              <div className="mb-4">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-700">
                  Tiếng Địa Phương
                </span>
              </div>
              <textarea
                className="flex-1 w-full bg-transparent resize-none outline-none text-gray-800 text-lg placeholder:text-gray-300"
                placeholder="Nhập câu phương ngữ của bạn ở đây..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>

            {/* Output Section */}
            <div className="flex-1 p-6 flex flex-col bg-gray-50/50 rounded-r-2xl pb-16">
              <div className="mb-4">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-700">
                  Tiếng Phổ Thông
                </span>
              </div>
              <div className="flex-1 w-full text-lg overflow-y-auto whitespace-pre-wrap text-gray-800">
                {result ? result : <span className="text-gray-400 italic">Bản dịch sẽ xuất hiện tại đây...</span>}
              </div>
            </div>
          </div>

          {/* Translate Button centrally overlapping */}
          <div className="absolute left-1/2 bottom-0 transform -translate-x-1/2 translate-y-1/2">
            <button
              onClick={handleTranslate}
              disabled={loading}
              className="bg-[#0055ff] hover:bg-blue-700 text-white px-8 py-3.5 rounded-full font-medium text-base shadow-lg transition-all flex items-center gap-2 focus:outline-none focus:ring-4 focus:ring-blue-300 disabled:opacity-70 disabled:cursor-not-allowed"
            >
              <Sparkles className="w-5 h-5" />
              {loading ? "Đang dịch..." : "Dịch ngay"}
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-4 px-8 mt-12 flex items-center justify-between z-10">
        <div className="w-8 h-8 bg-[#333] rounded-full flex items-center justify-center text-white text-xs font-bold font-mono">
          N
        </div>
        <p className="text-sm text-gray-500">
          STATE OF THE ART DỊCH PHƯƠNG NGỮ VIỆT NAM
        </p>
        <div className="w-8"></div> {/* Spacer for center alignment */}
      </footer>
    </div>
  );
}
