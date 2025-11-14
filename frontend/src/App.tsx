import { Routes, Route, Navigate } from 'react-router-dom'
import { Suspense, lazy } from 'react'

import { Layout } from '@/components/Layout'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { ErrorBoundary } from '@/components/ErrorBoundary'

// Lazy load components for code splitting
const HomePage = lazy(() => import('@/pages/HomePage'))
const GameGenerator = lazy(() => import('@/pages/GameGenerator'))
const GameLibrary = lazy(() => import('@/pages/GameLibrary'))
const GameViewer = lazy(() => import('@/pages/GameViewer'))
const PlanetInterface = lazy(() => import('@/pages/PlanetInterface'))
const ProfilePage = lazy(() => import('@/pages/ProfilePage'))
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'))

function App() {
  return (
    <ErrorBoundary>
      <Layout>
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            {/* Public routes */}
            <Route path="/" element={<HomePage />} />
            <Route path="/planet" element={<PlanetInterface />} />

            {/* Game-related routes */}
            <Route path="/generate" element={<GameGenerator />} />
            <Route path="/games" element={<GameLibrary />} />
            <Route path="/games/:gameId" element={<GameViewer />} />

            {/* User routes */}
            <Route path="/profile" element={<ProfilePage />} />

            {/* Fallback routes */}
            <Route path="/home" element={<Navigate to="/" replace />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </Layout>
    </ErrorBoundary>
  )
}

export default App