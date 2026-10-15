import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "U.S. Electrolysis Cost Explorer",
  description:
    "State-resolved plant-gate electrolysis cost screening based on the revised deterministic v3 analysis.",
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
