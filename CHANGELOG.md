# Changelog

## [0.3.0](https://github.com/yildirimarda/pageloom/compare/v0.2.0...v0.3.0) (2026-10-11)


### Features

* wire ConsistencyGuard into the generation path with vision_analyzer ([#64](https://github.com/yildirimarda/pageloom/issues/64)) ([e1d96b4](https://github.com/yildirimarda/pageloom/commit/e1d96b4951619ec71ae4b56aac02a2e32b446ee6))


### Bug Fixes

* correct lmstudio model id to match LM Studio's actual /v1/models output (qwen/qwen3.8-27b) ([ee8e6b4](https://github.com/yildirimarda/pageloom/commit/ee8e6b4f13eabb4b6aead635817b7414be046d55))
* load Stable Diffusion pipeline with safety_checker=None at from_pretrained ([#78](https://github.com/yildirimarda/pageloom/issues/78)) ([2718ce7](https://github.com/yildirimarda/pageloom/commit/2718ce7fa587e99126be7a27cc09677adaef9dcc))
* pin CPU-only torch on Linux to stop CUDA-lib segfault loop ([3500fd1](https://github.com/yildirimarda/pageloom/commit/3500fd147fb9715fd97c694d7483d22eb4a69bbc))
* round SD generation dimensions to multiples of 8 before diffusers calls ([#63](https://github.com/yildirimarda/pageloom/issues/63)) ([d459528](https://github.com/yildirimarda/pageloom/commit/d45952853b5101a665aead34e97c256f04defae6))

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
