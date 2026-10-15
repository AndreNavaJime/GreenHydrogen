"use client";

import { useEffect, useState } from "react";
import { assetUrl } from "@/lib/csv";

type InstallChoice = { outcome: "accepted" | "dismissed"; platform: string };

interface InstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<InstallChoice>;
}

type NavigatorStandalone = Navigator & { standalone?: boolean };

export default function PWAInstall() {
  const [promptEvent, setPromptEvent] = useState<InstallPromptEvent | null>(null);
  const [isIOS, setIsIOS] = useState(false);
  const [isStandalone, setIsStandalone] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register(assetUrl("/sw.js"), {
        scope: assetUrl("/") || "/",
      }).catch(() => undefined);
    }

    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      (navigator as NavigatorStandalone).standalone === true;

    const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
    const mobileLike =
      window.matchMedia("(max-width: 860px)").matches ||
      window.matchMedia("(pointer: coarse)").matches;

    setIsStandalone(standalone);
    setIsIOS(ios);

    const dismissed = sessionStorage.getItem("hydrogen-install-dismissed") === "1";
    if (!standalone && mobileLike && !dismissed && ios) {
      setVisible(true);
    }

    const onPrompt = (event: Event) => {
      event.preventDefault();
      const installEvent = event as InstallPromptEvent;
      setPromptEvent(installEvent);
      if (!standalone && mobileLike && !dismissed) {
        setVisible(true);
      }
    };

    const onInstalled = () => {
      setVisible(false);
      setPromptEvent(null);
      setIsStandalone(true);
    };

    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);

    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  const install = async () => {
    if (!promptEvent) return;
    await promptEvent.prompt();
    const choice = await promptEvent.userChoice;
    if (choice.outcome === "accepted") {
      setVisible(false);
    }
    setPromptEvent(null);
  };

  const dismiss = () => {
    sessionStorage.setItem("hydrogen-install-dismissed", "1");
    setVisible(false);
  };

  if (!visible || isStandalone) return null;

  return (
    <aside className="install-card" aria-live="polite">
      <button className="install-close" onClick={dismiss} aria-label="Close install prompt">×</button>
      <div className="install-mark" aria-hidden="true"><span /><i /><b /></div>
      <div className="install-copy">
        <strong>Install Hydrogen USA</strong>
        {isIOS ? (
          <span>On iPhone: tap Share, then <b>Add to Home Screen</b>.</span>
        ) : (
          <span>Add this explorer to your phone and open it like an app.</span>
        )}
      </div>
      {!isIOS && promptEvent && (
        <button className="install-button" onClick={install}>Install</button>
      )}
    </aside>
  );
}
