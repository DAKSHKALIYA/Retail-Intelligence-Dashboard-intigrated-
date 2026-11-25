package com.retail.api.service;

import com.retail.api.model.CustomerScore;
import com.retail.api.repo.CustomerScoreRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** ETL step: loads the Python-generated ml_output/customer_scores.csv into the database at startup. */
@Component
public class CsvLoader implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(CsvLoader.class);

    private final CustomerScoreRepository repo;
    private final String csvPath;

    public CsvLoader(CustomerScoreRepository repo, @Value("${scores.csv.path}") String csvPath) {
        this.repo = repo;
        this.csvPath = csvPath;
    }

    @Override
    public void run(ApplicationArguments args) throws IOException {
        Path path = Path.of(csvPath).toAbsolutePath().normalize();
        if (!Files.exists(path)) {
            log.error("Scores file not found: {}. Run 'python ml_models.py' first, "
                    + "or set scores.csv.path.", path);
            return;
        }
        List<String> lines = Files.readAllLines(path);
        String[] header = lines.get(0).trim().split(",");
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            idx.put(header[i], i);
        }

        List<CustomerScore> batch = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isBlank()) continue;
            String[] c = lines.get(i).trim().split(",");
            batch.add(new CustomerScore(
                    (long) Double.parseDouble(c[idx.get("CustomerID")]),
                    (int) Double.parseDouble(c[idx.get("Recency")]),
                    (int) Double.parseDouble(c[idx.get("Frequency")]),
                    Double.parseDouble(c[idx.get("Monetary")]),
                    c[idx.get("Segment")],
                    c[idx.get("ClusterName")],
                    Double.parseDouble(c[idx.get("ChurnProb")]),
                    c[idx.get("ChurnRisk")],
                    Double.parseDouble(c[idx.get("PredCLV90")])));
        }
        repo.saveAll(batch);
        log.info("Loaded {} customers from {}", batch.size(), path);
    }
}
