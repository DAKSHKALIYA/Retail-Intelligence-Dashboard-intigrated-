package com.retail.api.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

/** Calls the Python FastAPI ML service. */
@Service
public class MlServiceClient {

    private final RestClient client;

    public MlServiceClient(@Value("${python.service.url}") String baseUrl) {
        this.client = RestClient.create(baseUrl);
    }

    public Map<String, Object> health() {
        return call("/health");
    }

    public Map<String, Object> customer(long id) {
        return call("/customers/" + id);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> call(String path) {
        try {
            return client.get().uri(path).retrieve().body(Map.class);
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "Python ML service unavailable: " + e.getMessage());
        }
    }
}
