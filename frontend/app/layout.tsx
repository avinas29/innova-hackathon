import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "VERITAS — Multi-Agent Research & Fact Verification",
  description:
    "Every claim decomposed, verified against independent sources, and scored with calibrated confidence.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
