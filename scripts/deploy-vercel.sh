#!/bin/bash
# Vercel deployment script
set -e
echo "🚀 Deploying to Vercel..."
# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi
# Build if not already built
if [ ! -d "dist" ]; then
    echo "🔨 Building frontend..."
    npm run build
fi
# Deploy to Vercel
if [ "$NODE_ENV" = "production" ]; then
    echo "🌐 Deploying to production..."
    vercel --token "$VERCEL_TOKEN" --prod --confirm dist
else
    echo "🔍 Deploying to preview..."
    vercel --token "$VERCEL_TOKEN" --confirm dist
fi
echo "✅ Deployment complete!"
