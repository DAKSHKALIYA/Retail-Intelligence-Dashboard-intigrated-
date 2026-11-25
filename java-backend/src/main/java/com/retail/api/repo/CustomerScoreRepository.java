package com.retail.api.repo;

import com.retail.api.dto.SegmentSummary;
import com.retail.api.model.CustomerScore;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface CustomerScoreRepository extends JpaRepository<CustomerScore, Long> {

    List<CustomerScore> findBySegment(String segment, Pageable pageable);

    List<CustomerScore> findByChurnRisk(String churnRisk, Pageable pageable);

    List<CustomerScore> findBySegmentAndChurnRisk(String segment, String churnRisk, Pageable pageable);

    @Query("""
            select new com.retail.api.dto.SegmentSummary(
                c.segment, count(c), sum(c.monetary), avg(c.churnProb), sum(c.predClv90))
            from CustomerScore c
            group by c.segment
            order by sum(c.monetary) desc
            """)
    List<SegmentSummary> segmentSummary();
}
