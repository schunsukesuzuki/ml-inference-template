package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Backend represents a worker backend
type Backend struct {
	URL          *url.URL
	Alive        bool
	ReverseProxy *httputil.ReverseProxy
	mux          sync.RWMutex
}

// SetAlive updates the backend status
func (b *Backend) SetAlive(alive bool) {
	b.mux.Lock()
	b.Alive = alive
	b.mux.Unlock()
}

// IsAlive returns the backend status
func (b *Backend) IsAlive() bool {
	b.mux.RLock()
	alive := b.Alive
	b.mux.RUnlock()
	return alive
}

// LoadBalancer manages multiple backends
type LoadBalancer struct {
	backends []*Backend
	current  uint64
}

// getNextBackend returns the next alive backend using round-robin
func (lb *LoadBalancer) getNextBackend() *Backend {
	// Try to find an alive backend
	for i := 0; i < len(lb.backends)*2; i++ {
		idx := int(atomic.AddUint64(&lb.current, 1)) % len(lb.backends)
		if lb.backends[idx].IsAlive() {
			return lb.backends[idx]
		}
	}
	return nil
}

// ServeHTTP handles incoming requests and forwards to backends
func (lb *LoadBalancer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	backend := lb.getNextBackend()
	
	if backend == nil {
		http.Error(w, "Service unavailable: all backends are down", http.StatusServiceUnavailable)
		log.Printf("All backends are down for request: %s %s", r.Method, r.URL.Path)
		return
	}
	
	log.Printf("Routing %s %s to %s", r.Method, r.URL.Path, backend.URL)
	backend.ReverseProxy.ServeHTTP(w, r)
}

// healthCheck periodically checks backend health
func (lb *LoadBalancer) healthCheck(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			log.Println("Health check stopped")
			return
		case <-ticker.C:
			for _, backend := range lb.backends {
				go func(b *Backend) {
					client := &http.Client{
						Timeout: 3 * time.Second,
					}
					
					resp, err := client.Get(b.URL.String() + "/health")
					if err != nil || resp.StatusCode != 200 {
						if b.IsAlive() {
							log.Printf("Backend %s is DOWN", b.URL)
						}
						b.SetAlive(false)
					} else {
						if !b.IsAlive() {
							log.Printf("Backend %s is UP", b.URL)
						}
						b.SetAlive(true)
						resp.Body.Close()
					}
				}(backend)
			}
		}
	}
}

// loggingMiddleware logs all requests
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Create a response writer wrapper to capture status code
		wrapper := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(wrapper, r)
		
		duration := time.Since(start)
		log.Printf("%s %s - Status: %d - Duration: %v - RemoteAddr: %s",
			r.Method, r.URL.Path, wrapper.statusCode, duration, r.RemoteAddr)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func main() {
	// Get workers from environment variable
	workersEnv := os.Getenv("WORKERS")
	if workersEnv == "" {
		log.Fatal("WORKERS environment variable is not set")
	}
	
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	workers := strings.Split(workersEnv, ",")
	
	lb := &LoadBalancer{}
	
	// Initialize backends
	for _, worker := range workers {
		worker = strings.TrimSpace(worker)
		if worker == "" {
			continue
		}
		
		workerURL := worker
		if !strings.HasPrefix(worker, "http://") && !strings.HasPrefix(worker, "https://") {
			workerURL = "http://" + worker
		}
		
		parsedURL, err := url.Parse(workerURL)
		if err != nil {
			log.Printf("Invalid worker URL %s: %v", worker, err)
			continue
		}
		
		proxy := httputil.NewSingleHostReverseProxy(parsedURL)
		
		// Custom error handler
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("Error proxying to %s: %v", parsedURL, err)
			http.Error(w, "Bad Gateway", http.StatusBadGateway)
		}
		
		backend := &Backend{
			URL:          parsedURL,
			Alive:        true,
			ReverseProxy: proxy,
		}
		
		lb.backends = append(lb.backends, backend)
		log.Printf("Added backend: %s", parsedURL)
	}
	
	if len(lb.backends) == 0 {
		log.Fatal("No valid worker backends configured")
	}
	
	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// Start health checks
	go lb.healthCheck(ctx)
	
	// Setup HTTP server
	mux := http.NewServeMux()
	
	// Gateway health endpoint
	mux.HandleFunc("/gateway/health", func(w http.ResponseWriter, r *http.Request) {
		aliveCount := 0
		for _, b := range lb.backends {
			if b.IsAlive() {
				aliveCount++
			}
		}
		
		status := "healthy"
		statusCode := http.StatusOK
		
		if aliveCount == 0 {
			status = "unhealthy"
			statusCode = http.StatusServiceUnavailable
		} else if aliveCount < len(lb.backends) {
			status = "degraded"
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		fmt.Fprintf(w, `{"status":"%s","alive_backends":%d,"total_backends":%d}`, 
			status, aliveCount, len(lb.backends))
	})
	
	// Gateway info endpoint
	mux.HandleFunc("/gateway/info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"gateway":"ml-inference-gateway","version":"1.0.0","backends":%d}`, 
			len(lb.backends))
	})
	
	// All other requests go to load balancer
	mux.Handle("/", lb)
	
	// Wrap with logging middleware
	handler := loggingMiddleware(mux)
	
	// Create server
	server := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	
	// Handle graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		
		log.Println("Shutting down gateway...")
		cancel() // Stop health checks
		
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()
		
		if err := server.Shutdown(shutdownCtx); err != nil {
			log.Printf("Error during shutdown: %v", err)
		}
	}()
	
	// Start server
	log.Printf("Gateway listening on :%s with %d backends", port, len(lb.backends))
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
	
	log.Println("Gateway stopped")
}
