package com.retail.api.dto;

public record SegmentSummary(
        String segment,
        long customers,
        double totalRevenue,
        double avgChurnProb,
        double predictedClv90) {}
