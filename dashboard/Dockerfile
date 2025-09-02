# dashboard/Dockerfile

# Base stage with common setup
FROM node:22-slim AS base
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
EXPOSE 3000

# Installer stage for production dependencies
FROM base AS installer
COPY package.json package-lock.json ./
RUN npm ci
RUN npm install --save-exact --save-dev typescript
COPY . .

# Development stage (conditional install via entrypoint)
FROM base AS dev
ENV NODE_ENV=development
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["npm", "run", "dev"]

# Production stage
FROM base AS prod
ENV NODE_ENV=production
COPY --from=installer /app /app
RUN npm run build
CMD ["npm", "run", "start"]