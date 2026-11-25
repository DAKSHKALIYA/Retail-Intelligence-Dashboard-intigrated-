package com.retail.api.web;

import com.retail.api.dto.SegmentSummary;
import com.retail.api.model.CustomerScore;
import com.retail.api.repo.CustomerScoreRepository;
import com.retail.api.service.MlServiceClient;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class CustomerController {

    private final CustomerScoreRepository repo;
    private final MlServiceClient ml;

    public CustomerController(CustomerScoreRepository repo, MlServiceClient ml) {
        this.repo = repo;
        this.ml = ml;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "ok", "customers", repo.count());
    }

    /** GET /api/customers?segment=Champions&risk=High&limit=100 */
    @GetMapping("/customers")
    public List<CustomerScore> customers(@RequestParam(required = false) String segment,
                                         @RequestParam(required = false) String risk,
                                         @RequestParam(defaultValue = "1000") int limit) {
        Pageable page = PageRequest.of(0, Math.max(1, limit), Sort.by("customerId"));
        if (segment != null && risk != null) return repo.findBySegmentAndChurnRisk(segment, risk, page);
        if (segment != null) return repo.findBySegment(segment, page);
        if (risk != null) return repo.findByChurnRisk(risk, page);
        return repo.findAll(page).getContent();
    }

    @GetMapping("/customers/{id}")
    public CustomerScore customer(@PathVariable long id) {
        return repo.findById(id).orElseThrow(() ->
                new ResponseStatusException(HttpStatus.NOT_FOUND, "Customer " + id + " not found"));
    }

    /** Highest-spending customers who are likely to churn. */
    @GetMapping("/at-risk")
    public List<CustomerScore> atRisk(@RequestParam(defaultValue = "20") int limit) {
        return repo.findByChurnRisk("High", PageRequest.of(0, Math.max(1, limit), Sort.by(Sort.Direction.DESC, "monetary")));
    }

    @GetMapping("/segments")
    public List<SegmentSummary> segments() {
        return repo.segmentSummary();
    }

    // ---- calls into the Python service --------------------------------------------------
    @GetMapping("/ml/health")
    public Map<String, Object> mlHealth() {
        return ml.health();
    }

    @GetMapping("/customers/{id}/ml")
    public Map<String, Object> customerFromMl(@PathVariable long id) {
        return ml.customer(id);
    }
}
