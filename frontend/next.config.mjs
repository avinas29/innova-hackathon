/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Static export: the whole UI is client-side ("use client", fetch + SSE), so
  // there is nothing for a Node server to render. Exporting to plain files lets
  // FastAPI serve the UI and the API from one process on one port — which means
  // one Render service, one domain, and no CORS at all.
  output: "export",

  // Required by `output: export` — there is no server to optimise images.
  images: { unoptimized: true },

  // Emit `about/index.html` rather than `about.html`, so static hosts resolve
  // extensionless paths correctly.
  trailingSlash: true,
};

export default nextConfig;
