import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Dịch Phương Ngữ Demo",
  description: "AI hỗ trợ chuyển đổi ngôn ngữ vùng miền",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body className={`${inter.className} bg-[#f8f9fc] text-gray-900 antialiased`}>
        {children}
      </body>
    </html>
  );
}
