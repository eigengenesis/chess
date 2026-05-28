# Chess Hosting Steps

The clean setup is:

1. Cloudflare Pages hosts `frontend/`.
2. A private Python backend hosts the PyTorch model.
3. The Pages Function in `frontend/functions/api/[[path]].js` proxies `/api/*` to that backend.

That keeps `chess_god.pt` and `chess_god_mode.py` off the public website.

## 1. Run It Locally

From this folder:

```bash
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate ml-env
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
python -m http.server 5173 -d frontend
```

Open:

```text
http://localhost:5173
```

Check the backend:

```bash
curl http://localhost:8000/api/health
```

Test one model move:

```bash
curl -X POST http://localhost:8000/api/move \
  -H "Content-Type: application/json" \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

## 2. Keep The Model Private

Do not deploy these to Cloudflare Pages:

- `chess_god.pt`
- `chess_god_mode.py`
- `backend/`
- training scripts

The `.gitignore` already excludes checkpoint files. If you upload the frontend manually, upload only the `frontend/` directory.

Optional at-rest checkpoint encryption:

```bash
pip install cryptography
python tools/encrypt_checkpoint.py chess_god.pt chess_god.pt.enc
```

On the backend host, set:

```text
CHESS_GOD_CHECKPOINT=chess_god.pt.enc
CHESS_GOD_ENCRYPTED=1
CHESS_GOD_KEY=<the printed key>
```

This protects the file at rest. It does not hide the model from the backend machine, because the backend must decrypt the checkpoint to run inference.

## 3. Choose The Backend Route

### Free Route: Your Machine + Cloudflare Tunnel

This is the easiest free option, but your computer must stay awake and connected.

Run the backend locally:

```bash
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate ml-env
API_SHARED_SECRET="make-a-long-random-secret" \
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Create a Cloudflare Tunnel that points a subdomain like `api.yourdomain.com` to:

```text
http://localhost:8000
```

Then your private API base becomes:

```text
https://api.yourdomain.com
```

Dashboard steps:

1. Add your domain to Cloudflare and make sure it uses Cloudflare nameservers.
2. Go to `Zero Trust -> Networks -> Tunnels`.
3. Create a tunnel named `chess-god-api`.
4. Choose `cloudflared`.
5. Copy and run the install/run command Cloudflare gives you.
6. Add a public hostname:
   - Subdomain: `api`
   - Domain: `yourdomain.com`
   - Type: `HTTP`
   - URL: `localhost:8000`
7. Save, then open `https://api.yourdomain.com/api/health`.

### Production Route: Python/Container Host

Use any host that can run Python + PyTorch. Deploy this project privately and start:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

Set these backend environment variables:

```text
API_SHARED_SECRET=make-a-long-random-secret
CHESS_GOD_CHECKPOINT=/path/to/chess_god.pt
TORCH_NUM_THREADS=1
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## 4. Deploy The Frontend To Cloudflare Pages

Because this project uses a `functions/` folder, deploy with Wrangler:

```bash
cd frontend
npx wrangler login
npx wrangler pages project create chess-god
npx wrangler pages deploy . --project-name chess-god
```

Cloudflare will give you a `*.pages.dev` URL.

## 5. Add Pages Environment Variables

In Cloudflare dashboard:

```text
Workers & Pages -> your Pages project -> Settings -> Variables and Secrets
```

Add:

```text
MODEL_API_BASE=https://api.yourdomain.com
MODEL_API_SECRET=the-same-value-as-API_SHARED_SECRET
```

Redeploy after adding them:

```bash
cd frontend
npx wrangler pages deploy . --project-name chess-god
```

## 6. Connect Your Domain

In Cloudflare dashboard:

```text
Workers & Pages -> your Pages project -> Custom domains -> Set up a domain
```

Add:

```text
yourdomain.com
www.yourdomain.com
```

If the domain is already using Cloudflare nameservers, Cloudflare usually creates the DNS records for you. If not, add the CNAME that Cloudflare shows you.

## 7. Final Test

Visit:

```text
https://yourdomain.com/api/health
```

You should see JSON from the backend through Cloudflare. Then open:

```text
https://yourdomain.com
```

Start a game. The top-right status should say `Ready`; during model inference it should say `Thinking`.
