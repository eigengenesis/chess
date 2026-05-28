export async function onRequest({ request, env, params }) {
  const origin = request.headers.get("Origin") || new URL(request.url).origin;
  const cors = {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    Vary: "Origin",
  };

  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: cors });
  }

  if (!env.MODEL_API_BASE) {
    return Response.json(
      { detail: "MODEL_API_BASE is not configured." },
      { status: 503, headers: cors },
    );
  }

  const path = Array.isArray(params.path) ? params.path.join("/") : params.path || "";
  const incoming = new URL(request.url);
  const target = new URL(`${env.MODEL_API_BASE.replace(/\/$/, "")}/api/${path}`);
  target.search = incoming.search;

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("cookie");
  headers.delete("authorization");
  if (env.MODEL_API_SECRET) {
    headers.set("X-API-Key", env.MODEL_API_SECRET);
  }
  if (env.HF_TOKEN) {
    headers.set("Authorization", `Bearer ${env.HF_TOKEN}`);
  }

  const response = await fetch(target, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "manual",
  });

  const responseHeaders = new Headers(response.headers);
  Object.entries(cors).forEach(([key, value]) => responseHeaders.set(key, value));
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}
