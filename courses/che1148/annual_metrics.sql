-- Denominator is customers with an observed spend value in each year/response segment.
SELECT year, response,
       COUNT(annual_spend) AS customer_count,
       ROUND(AVG(annual_spend), 2) AS mean_annual_spend,
       ROUND(SUM(annual_spend), 2) AS total_annual_spend
FROM customer_year
WHERE annual_spend IS NOT NULL
GROUP BY year, response
ORDER BY year, response;
