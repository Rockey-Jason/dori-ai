# Dori AI site integration

## Render environment
Set these environment variables on the Dori AI Render service:
- `SUPABASE_URL` = the Doldol site's Supabase project URL
- `SUPABASE_ANON_KEY` = the Supabase anon/public key
- `DORI_WEB_SEARCH=1` (optional; enabled by default)

## Logged-in user ID
The chat request accepts `user_id`. The bundled frontend sends `window.DORI_USER_ID` first, then `localStorage.dori_user_id`.

For the real Doldol site, set `window.DORI_USER_ID` from the currently authenticated Supabase user's `id`/`sub` **after authentication**. Do not hard-code another user's ID.

The backend uses that ID to read only `users.read_dori_news`. If a requested newspaper number is higher than that limit, its contents are not returned.

## Important security note
This package's current compatibility path trusts the supplied `user_id`; it is not a replacement for JWT verification. For a hardened production deployment, send the Supabase access token in `Authorization: Bearer ...` and derive the user identity on the server before reading `read_dori_news`.

## Current data features
- multilingual lightweight detection/normalization
- Dori newspaper lookup
- per-user readable-news limit
- newspaper quiz/question display without revealing the answer key
- current active Doldol Securities data from Supabase
- local Dori knowledge retrieval
- optional public-web source finding
- from-scratch NumPy Transformer fallback
