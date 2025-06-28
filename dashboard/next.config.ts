import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  productionBrowserSourceMaps: false,
  assetPrefix: process.env.NODE_ENV === 'production' ? '.' : '',
  output: 'export',
};

export default nextConfig;
