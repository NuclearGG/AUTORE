---
title: AUTORE Environment Server
emoji: \U0001f6a6
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AUTORE - Autonomous Traffic Optimization & Response Engine

A reinforcement learning environment for intelligent traffic signal control at a four-way intersection.

## Observation Space (6-dim)
- `cars_N`, `cars_S`, `cars_E`, `cars_W` - vehicles queued per lane
- `phase` - current signal (0=NS green, 1=EW green, -1=yellow)
- `emergency_lane` - ambulance location (-1 if none, 0-3 for lane)

## Action Space (Discrete 2)
- `0` North-South green
- `1` East-West green

## Reward
- `-total_waiting_cars` per step
- x3 multiplier during yellow transition
- `-50` per step while ambulance is blocked

## Real-World Layers
- Asymmetric traffic: NS arterial (rate=3.0) vs EW side roads (rate=1.2)
- Rush hour: x2.5 spike during steps 40-80 of each episode
- Emergency vehicles: 4% spawn probability per step

## API Endpoints
- `POST /reset` - reset environment
- `POST /step` - body: `{"signal": 0}` or `{"signal": 1}`
- `GET /state` - current state
- `GET /health` - health check
- `GET /docs` - OpenAPI docs
- `/web` - interactive web UI

## Tasks and Graders
| Task | Objective | Score |
|------|-----------|-------|
| easy | Minimize average waiting cars | [0, 1] |
| medium | Handle rush-hour congestion | [0, 1] |
| hard | Prioritize emergency vehicles | [0, 1] |

## Run Inference
```bash
python inference.py
USE_LLM=1 python inference.py
```

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| API_BASE_URL | LLM endpoint | https://api.openai.com/v1 |
| MODEL_NAME | Model identifier | gpt-4o-mini |
| HF_TOKEN | API key | (required for LLM mode) |
