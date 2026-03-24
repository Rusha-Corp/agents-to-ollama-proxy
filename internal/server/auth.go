package server

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

func requireBearerAuth(token string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization := r.Header.Get("Authorization")
		providedToken, ok := strings.CutPrefix(authorization, "Bearer ")
		if !ok || subtle.ConstantTimeCompare([]byte(providedToken), []byte(token)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="ollama-proxy"`)
			writeError(w, http.StatusUnauthorized, "unauthorized", "authentication_error", "unauthorized")
			return
		}
		next.ServeHTTP(w, r)
	})
}
