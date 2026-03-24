#!/bin/bash
cd /www/models-varroa-on-bee/

COMPOSE_PROJECT_NAME=gratheon docker-compose -f docker-compose.prod.yml down
COMPOSE_PROJECT_NAME=gratheon docker-compose -f docker-compose.prod.yml up --build -d
