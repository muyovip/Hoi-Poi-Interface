# CapsuleOS Game Generator Frontend

Modern React frontend for the multi-LLM game generation system powered by CapsuleOS immutable graph patterns.

## 🚀 Features

- **Modern React Architecture**: Built with React 18, TypeScript, and modern development tools
- **Responsive Design**: Mobile-first design with Tailwind CSS and glassmorphism effects
- **3D Visualization**: Three.js integration for planetary interface and game visualization
- **Dark Mode**: Full dark mode support with system preference detection
- **Real-time Updates**: React Query for efficient data fetching and caching
- **Form Management**: React Hook Form with Zod validation
- **Animations**: Framer Motion for smooth, interactive animations
- **Type Safety**: Full TypeScript implementation with strict type checking

## 🎯 Core Pages

- **Home Page**: Hero section with feature highlights and call-to-action
- **Game Generator**: Interactive form for creating new games with real-time progress
- **Game Library**: Browse and manage your collection of generated games
- **Game Viewer**: Detailed view and interaction with generated games
- **Planet Interface**: 3D planetary visualization of your game universe
- **Profile Page**: User account management and settings

## 🛠️ Tech Stack

- **Frontend Framework**: React 18 with functional components and hooks
- **Build Tool**: Vite for lightning-fast development and building
- **Styling**: Tailwind CSS with custom design system and dark mode
- **Type Checking**: TypeScript 5 with strict mode enabled
- **Routing**: React Router DOM with lazy loading and code splitting
- **State Management**: Zustand for global state and React Query for server state
- **Forms**: React Hook Form with Zod validation and error handling
- **Animations**: Framer Motion for sophisticated UI animations
- **3D Graphics**: Three.js with React Three Fiber and Drei components
- **HTTP Client**: Axios for API communication with interceptors
- **Icons**: Heroicons for consistent icon system
- **Notifications**: React Hot Toast for toast notifications
- **Validation**: Zod for schema validation and type inference

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/           # Reusable UI components
│   │   ├── Layout.tsx      # Main app layout
│   │   ├── ui/              # Core UI components
│   │   │   ├── Logo.tsx
│   │   │   ├── LoadingSpinner.tsx
│   │   │   └── ErrorBoundary.tsx
│   │   └── ErrorBoundary.tsx
│   ├── hooks/               # Custom React hooks
│   │   ├── useDarkMode.ts
│   │   ├── useAuth.ts
│   │   └── useCapsule.ts      # TODO: Capsule management hook
│   ├── pages/               # Page components
│   │   ├── HomePage.tsx
│   │   ├── GameGenerator.tsx
│   │   ├── GameLibrary.tsx
│   │   ├── GameViewer.tsx
│   │   ├── PlanetInterface.tsx
│   │   ├── ProfilePage.tsx
│   │   └── NotFoundPage.tsx
│   ├── main.tsx             # App entry point
│   ├── App.tsx              # App component with routing
│   └── index.css            # Global styles and Tailwind imports
├── public/                  # Static assets
├── index.html              # HTML template
├── package.json            # Dependencies and scripts
├── vite.config.ts         # Vite configuration
├── tailwind.config.js     # Tailwind CSS configuration
├── tsconfig.json          # TypeScript configuration
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18.0 or higher
- npm, yarn, or pnpm package manager

### Installation

1. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

2. **Start development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

3. **Open your browser**
   Navigate to `http://localhost:3000`

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint to check code quality
- `npm run lint:fix` - Fix ESLint issues automatically
- `npm run type-check` - Run TypeScript type checking
- `npm run test` - Run Vitest unit tests

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:9000
VITE_CAPSULOS_API_URL=http://localhost:8080
VITE_VLLM_API_URL=http://localhost:8000

# Application Settings
VITE_APP_NAME="CapsuleOS Game Generator"
VITE_APP_DESCRIPTION="Multi-LLM game generation system"
VITE_ENABLE_ANALYTICS=false
VITE_ENVIRONMENT=development
```

### Development

- **Hot Module Replacement (HMR)**: Changes are reflected instantly
- **TypeScript**: Full type safety with strict mode
- **ESLint**: Code quality enforcement with React-specific rules
- **Prettier**: Code formatting (configured in .prettierrc)
- **Git Hooks**: Pre-commit hooks for code quality

## 🎨 Design System

### Color Palette

- **Primary**: Blue gradient (#3b82f6 → #22c55e)
- **Secondary**: Green gradient (#4ade80 → #22c55e)
- **Background**: Light/dark adaptive
- **Text**: High contrast for accessibility

### Typography

- **Primary**: Inter (system font stack)
- **Code**: JetBrains Mono (for code and technical content)
- **Responsive**: Scale based on viewport size

### Components

- **Glassmorphism**: Frosted glass effect for modern UI
- **Cards**: Elevated content areas with hover effects
- **Forms**: Accessible form controls with validation
- **Buttons**: Primary, secondary, and outline variants
- **Loading**: Animated loading states with progress indicators

## 🌐 Backend Integration

### API Bridge

The frontend communicates with:

1. **CapsuleOS API** (`/api/v1/`)
   - Game storage and retrieval
   - Capsule management
   - Genesis graph operations

2. **vLLM API** (`/v1/`)
   - Game generation requests
   - LLM orchestration
   - Progress tracking

3. **Authentication Service**
   - User authentication and session management
   - Token-based security

### Real-time Features

- **WebSockets**: For real-time generation progress updates
- **Server-Sent Events**: For streaming generation results
- **Polling**: Fallback for real-time updates

## 📱 Responsive Design

### Breakpoints

- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Adaptive Features

- **Touch Gestures**: Swipe navigation and touch-optimized controls
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: ARIA labels and semantic HTML
- **Reduced Motion**: Respect user's motion preferences

## 🔒 Security Features

- **CORS**: Properly configured for cross-origin requests
- **CSRF Protection**: Token-based authentication
- **XSS Prevention**: Content Security Policy and input sanitization
- **HTTPS Enforcement**: Production-only HTTPS
- **Content Security**: Secure headers and content validation

## 🚀 Performance Optimizations

- **Code Splitting**: Route-based and component-level splitting
- **Lazy Loading**: Components and pages loaded on demand
- **Image Optimization**: WebP and responsive images
- **Caching**: Aggressive caching with cache invalidation
- **Bundle Analysis**: Visualized bundle size and optimization suggestions

## ♿ Accessibility

- **WCAG 2.1**: Compliance with web accessibility standards
- **ARIA Roles**: Proper semantic HTML and ARIA labels
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader**: Optimized for assistive technologies
- **Color Contrast**: High contrast ratios for readability
- **Focus Management**: Visible focus indicators and logical tab order

## 🧪 Testing

### Unit Testing

```bash
npm run test
```

### E2E Testing (Coming Soon)

```bash
npm run test:e2e
```

### Coverage

```bash
npm run test:coverage
```

## 🔮 Future Enhancements

### Phase 1 (Complete)
- [ ] GΛLYPH parser WASM compilation
- [ ] `useCapsule` React hook implementation
- [ ] 3D planetary interface completion
- [ ] Authentication integration
- [ ] End-to-end flow testing

### Phase 2 (Future)
- [ ] Real-time collaboration features
- [ ] Advanced 3D game visualization
- [ ] Game templates and presets
- [ ] Community features and sharing
- [ ] Advanced analytics and insights

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with proper commit messages
4. **Run tests** and ensure code quality
5. **Submit a pull request** with detailed description

## 📄 License

This project is part of the CapsuleOS ecosystem and follows the same licensing terms.

## 🆘 Support

For issues and questions:
- **Documentation**: Check the project wiki and API docs
- **Issues**: Create an issue on the repository
- **Discussions**: Join community discussions for general questions
- **Discord**: Join our Discord server for real-time support