import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import PWAInstall from "@/components/PWAInstall";
import "./globals.css";

export const metadata: Metadata = {
  title: "U.S. Electrolysis Cost Explorer",
  description:
    "State-resolved plant-gate electrolysis cost screening based on the revised deterministic v3 analysis.",
  manifest: "/manifest.webmanifest",
  applicationName: "Hydrogen USA",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Hydrogen USA",
  },
  formatDetection: {
    telephone: false,
  },
  icons: {
    icon: "/hydrogen-icon.svg",
    apple: "/hydrogen-icon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#12343B",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        {children}
        <PWAInstall />
      </body>
    </html>
  );
}
