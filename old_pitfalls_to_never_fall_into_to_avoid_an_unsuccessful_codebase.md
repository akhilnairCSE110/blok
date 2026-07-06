# Pitfalls

- Do not fake generated text.
- Do not call descriptor output progress.
- Do not add generic model support.
- Do not hide CPU/page-cache fallback.
- Do not stream every active expert every token if caching can avoid it.
