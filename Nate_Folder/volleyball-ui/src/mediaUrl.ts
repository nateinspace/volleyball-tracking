/** Resolves public-folder paths for Vite `base` (e.g. subpath deploys on Vercel). */
export function resolveMediaUrl(src: string): string {
  if (/^https?:\/\//i.test(src)) return src;
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  return `${base}${src.startsWith("/") ? src : `/${src}`}`;
}
