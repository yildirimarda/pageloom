# Changelog

## [0.2.0](https://github.com/yildirimarda/pageloom/compare/v0.1.0...v0.2.0) (2026-09-13)


### Features

* free, vision-based page-structure recognition (any page type) ([d713f42](https://github.com/yildirimarda/pageloom/commit/d713f42f40dbfa01c491708f3ebae705b07032b3))


### Bug Fixes

* **ci:** sync --frozen and uv run --no-sync so CI uses exactly the locked toolchain; log ruff version ([d583922](https://github.com/yildirimarda/pageloom/commit/d583922f0e050b64ee419ab943f4e16a0fe4e178))
* wire --description through CLI; remove vestigial alphabet-limit overflow bug ([b82c589](https://github.com/yildirimarda/pageloom/commit/b82c589ada20d2e03a8c1b432ae8522dd2fa7c4b))


### Documentation

* **plan:** confirm graceful non-vision-model fallback + log discovery/UX backlog ([14f01c9](https://github.com/yildirimarda/pageloom/commit/14f01c93ac9b1fbcaa969b6873e33da1f16d79dc))
* **plan:** log CI fix + vision-analysis hardening backlog (not yet implemented) ([455010a](https://github.com/yildirimarda/pageloom/commit/455010aecfcbcfacb2359d8c97989470f736ad67))

## 0.1.0 (2026-09-12)


### Features

* add pyproject.toml with diffusers, Pillow, pypdf, pytest, ruff; include MPS/CPU/CUDA backend config notes ([29b31ca](https://github.com/yildirimarda/pageloom/commit/29b31ca61b89eb03395216a329fed9c22a210f69))
