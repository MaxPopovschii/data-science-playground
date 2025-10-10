# Anomaly Detection in Network Traffic

Questo progetto si occupa di rilevamento anomalie nel traffico di rete, utile per cybersecurity e monitoraggio.

## Struttura del progetto
- `notebooks/` — Jupyter notebook per esplorazione, analisi e modellazione
- `data/` — Dati raw e preprocessati (es. KDD Cup, UNSW-NB15, CICIDS)
- `src/` — Script Python per preprocessing, training e valutazione
- `README.md` — Descrizione del progetto

## Obiettivi
- Analisi esplorativa del traffico di rete
- Preprocessing e feature engineering
- Addestramento di modelli di anomaly detection (Isolation Forest, Autoencoder, ecc.)
- Valutazione delle performance

## Link utili
- [KDD Cup 99 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/malmem-2017.html)

## Come iniziare
1. Scarica un dataset e inseriscilo nella cartella `data/`
2. Avvia i notebook per esplorare e modellare i dati
3. Consulta gli script in `src/` per pipeline automatizzate
