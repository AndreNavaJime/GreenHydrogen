const isPages = process.env.GITHUB_ACTIONS === "true";
const repoBase = "/GreenHydrogen";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath: isPages ? repoBase : "",
  assetPrefix: isPages ? repoBase + "/" : "",
};

export default nextConfig;
