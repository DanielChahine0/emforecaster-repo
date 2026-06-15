# Applications of Time Series Forecasting in Wireless Networks

Time series forecasting is used across many wireless tasks, from predicting network demand and performance to public-health applications. I grouped the work into three areas: (1) cellular traffic forecasting, (2) network KPI and quality-of-service (QoS) forecasting, and (3) electromagnetic field (EMF) exposure forecasting, and then covered a set of emerging physical-layer tasks. The models for each are listed in the tables. 
Most wireless forecasters are point predictors, the model outputs a single number per future step, scored by error metrics like RMSE and MAE, which shows how close that single number lands on the truth overall. Each subfield built out only the single trust property its consequences forced it to, and ignored the rest.

---

## Network KPI and Quality-of-Service Forecasting

This area forecasts the metrics that shape user experience and service guarantees: latency, throughput, resource-block use, loss rate, link speed, and overall QoS or QoE.

### More details about the KPI Forecasting Models (Table II)

---

## EMF Exposure Forecasting

EMF exposure forecasting predicts radio-frequency field strength or power over time and space. Its motivation is regulatory and public-health: networks must show they stay within exposure limits, and denser 5G and millimeter-wave deployments has raised concern about total exposure. 

### More details about the EMF Forecasting (Table III)


---

## Cellular Traffic Forecasting

Cellular traffic forecasting predicts how much demand (data, voice, or messaging) a cell or region will carry over time. The data is strongly seasonal, with daily and weekly cycles, correlated across close cells, and sensitive to instant bursts.

### More details about the Traffic Forecasting Models (Table IV)

---

## Other Areas: Beam Management, CSI, and WiFi Sensing

Beam management predicts the best beam direction before a user moves. Channel state information (CSI) forecasting predicts the channel in advance to offset feedback delay and channel aging in massive MIMO. WiFi sensing reuses channel measurements to detect human presence and activity.
