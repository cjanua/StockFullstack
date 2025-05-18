# Digital Transactions & Funds (DTF)

A fullstack portfolio management system built with Next.js and FastAPI, integrating with the Alpaca trading API.

![Dashboard Screenshot](docs/dashboard.png)

## Features

- 📈 Real-time portfolio tracking and visualization
- 🔍 Position management with advanced filtering
- 🤖 Portfolio optimization recommendations using mathematical models
- 📊 Historical performance analysis
- 💰 Order creation and management
- 🔐 User authentication and Alpaca API key management
- 📱 Responsive design with dark/light mode support

## Architecture

The application consists of three main components:

1. **Frontend**: Next.js application with React components
2. **Backend API**: FastAPI service for portfolio optimization
3. **CLI Tool**: Command-line interface for Alpaca API interactions

### System Architecture Diagram

```
┌───────────────┐     ┌───────────────┐     ┌────────────────┐
│  Next.js      │     │  FastAPI      │     │  Alpaca        │
│  Frontend     │────▶│  Portfolio    │────▶│  Trading API   │
│  (React)      │     │  Service      │     │                │
└───────────────┘     └───────────────┘     └────────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌────────────────┐
│  SQLite       │     │  Redis        │     │  Market        │
│  (Auth DB)    │     │  (Cache)      │     │  Data          │
└───────────────┘     └───────────────┘     └────────────────┘
```

## Getting Started

### Prerequisites

- Node.js >= 18.x
- Python >= 3.10
- Redis
- Alpaca API credentials

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/dtf.git
cd dtf
```

2. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your Alpaca API credentials
```

3. Install backend dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/alpaca/req.txt
```

4. Install trading-platform dependencies
```bash
cd trading-platform
npm install
```

5. Start Redis and Portfolio Service
```bash
podman-compose up -d
# or
docker-compose up -d
```

6. Run the development server
```bash
cd trading-platform
npm run dev
```

7. Visit `http://localhost:3000` in your browser

## Usage

### Web Interface

Navigate to `http://localhost:3000` to access the web interface. You'll need to:

1. Register a new account
2. Connect your Alpaca API credentials on the account page
3. Explore your portfolio data, positions, and orders

### CLI Tool

The CLI tool provides quick access to Alpaca API functions from the terminal:

```bash
# Get account information
./bin/run.sh trading/account

# Get positions
./bin/run.sh trading/account/positions

# Get portfolio history (last 30 days)
./bin/run.sh trading/account/history --days 30 --timeframe 1D
```

## API Documentation

### Portfolio Service API

The FastAPI portfolio service provides the following endpoints:

- `GET /health` - Health check endpoint
- `GET /api/portfolio/optimize` - Get optimized portfolio weights
- `GET /api/portfolio/recommendations` - Get buy/sell recommendations
- `GET /api/debug/history` - Debug endpoint for historical data
- `GET /api/debug/client` - Debug endpoint for client methods

### Next.js API Routes

The Next.js trading-platform provides these API routes:

- `/api/alpaca/account` - Get account information
- `/api/alpaca/positions` - Get positions
- `/api/alpaca/orders` - Manage orders
- `/api/alpaca/account/history` - Get portfolio history
- `/api/auth/*` - Authentication endpoints

## Project Structure

```
├── backend/
│   └── alpaca/
│       ├── cli/            # Command-line interface
│       ├── core/           # Core utilities and configuration
│       ├── handlers/       # API request handlers
│       ├── sdk/            # Alpaca SDK integration
│       ├── serializers/    # Data serialization
│       └── services/       # Portfolio optimization service
├── trading-platform/
│   ├── app/                # Next.js app directory
│   │   ├── api/            # API routes
│   │   └── */              # Page components
│   ├── components/         # React components
│   ├── hooks/              # Custom React hooks
│   ├── layouts/            # Page layouts
│   ├── lib/                # Utilities and API clients
│   ├── providers/          # React context providers
│   └── types/              # TypeScript type definitions
├── scripts/                # Utility scripts
└── podman-compose.yaml     # Container configuration
```

## Refactoring and Contribution

If you'd like to contribute to this project, please consider these areas for improvement:

1. **Testing**: Add unit and integration tests
2. **Documentation**: Improve inline documentation and API docs
3. **Performance**: Optimize data fetching and caching strategies
4. **Features**: Add more trading features and analytics

## License

[MIT License](LICENSE)