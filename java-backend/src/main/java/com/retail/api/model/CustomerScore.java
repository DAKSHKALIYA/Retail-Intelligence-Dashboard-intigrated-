package com.retail.api.model;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/** One row per customer: RFM values, segment, ML cluster, churn probability and predicted CLV. */
@Entity
@Table(name = "customer_score")
public class CustomerScore {

    @Id
    private Long customerId;
    private int recency;
    private int frequency;
    private double monetary;
    private String segment;
    private String clusterName;
    private double churnProb;
    private String churnRisk;
    private double predClv90;

    public CustomerScore() {}

    public CustomerScore(Long customerId, int recency, int frequency, double monetary, String segment,
                         String clusterName, double churnProb, String churnRisk, double predClv90) {
        this.customerId = customerId;
        this.recency = recency;
        this.frequency = frequency;
        this.monetary = monetary;
        this.segment = segment;
        this.clusterName = clusterName;
        this.churnProb = churnProb;
        this.churnRisk = churnRisk;
        this.predClv90 = predClv90;
    }

    public Long getCustomerId() { return customerId; }
    public int getRecency() { return recency; }
    public int getFrequency() { return frequency; }
    public double getMonetary() { return monetary; }
    public String getSegment() { return segment; }
    public String getClusterName() { return clusterName; }
    public double getChurnProb() { return churnProb; }
    public String getChurnRisk() { return churnRisk; }
    public double getPredClv90() { return predClv90; }
}
