package com.smartworkspace.booking.api;

import jakarta.validation.constraints.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class PredictController {

  private final WebClient webClient;

  public PredictController(@Value("${ml.base-url:http://localhost:8005}") String mlBaseUrl) {
    this.webClient = WebClient.builder().baseUrl(mlBaseUrl).build();
  }

  @PostMapping(value = "/predict", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
  public Mono<Map> predict(@RequestBody Map<String, Object> body) {
    return webClient.post()
        .uri("/predict")
        .contentType(MediaType.APPLICATION_JSON)
        .bodyValue(body)
        .retrieve()
        .bodyToMono(Map.class);
  }
}


