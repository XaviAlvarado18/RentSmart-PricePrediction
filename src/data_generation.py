# -*- coding: utf-8 -*-
"""
Generador de dataset ficticio de alquileres para Ciudad de Guatemala.

- Salida: data/raw/rent_guatemala.csv
- Objetivo (target): rent_price_gtq
- Relaciones:
  * Precio base por m² según zona (zonas premium más caras; ranking del usuario respetado).
  * Ajustes por tipo de propiedad, tamaño, antigüedad, amenities, estacionamientos, piso, vista, ruido,
    distancia a centros de negocio/transporte y seguridad del complejo.
  * Distribuciones diferentes por zona (p.ej., más apartamentos en 4/9/10/13/14/15; más casas en 2/5/6/7/11/12/17/18/CES).
Uso:
    python src/data_generation.py --n 5000 --seed 42
"""
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ZoneConfig:
    base_price_m2: float       # Precio base por m² en GTQ (ficticio pero razonable)
    weight: float              # Probabilidad de muestreo de la zona (se normaliza)
    apt_share: float           # Proporción de apartamentos vs casas (0-1)
    center_to_business_km: float  # Distancia típica a zonas de oficinas/servicios
    typical_security: float    # Probabilidad de que tenga seguridad 24/7
    typical_amenities: float   # Probabilidad de amenidades (gimnasio/piscina/área social)


# --- CONFIGURACIÓN DE ZONAS ---
# Nota: base_price_m2 respeta el ranking del usuario de "peor→mejor".
ZONES: Dict[str, ZoneConfig] = {
    # Centro corporativo/premium
    "Zona 14": ZoneConfig(base_price_m2=158, weight=0.09, apt_share=0.92, center_to_business_km=1.2, typical_security=0.95, typical_amenities=0.86),
    "Zona 10": ZoneConfig(base_price_m2=148, weight=0.11, apt_share=0.90, center_to_business_km=1.0, typical_security=0.90, typical_amenities=0.82),
    "Zona 15": ZoneConfig(base_price_m2=138, weight=0.09, apt_share=0.70, center_to_business_km=2.5, typical_security=0.86, typical_amenities=0.72),
    "Zona 16": ZoneConfig(base_price_m2=140, weight=0.08, apt_share=0.55, center_to_business_km=4.0, typical_security=0.82, typical_amenities=0.64),
    "Zona 9":  ZoneConfig(base_price_m2=120, weight=0.07, apt_share=0.85, center_to_business_km=1.5, typical_security=0.85, typical_amenities=0.75),
    "Zona 4":  ZoneConfig(base_price_m2=115, weight=0.06, apt_share=0.80, center_to_business_km=2.0, typical_security=0.80, typical_amenities=0.70),
    "Zona 13": ZoneConfig(base_price_m2=110, weight=0.05, apt_share=0.80, center_to_business_km=2.5, typical_security=0.80, typical_amenities=0.65),

    # Corredor hacia las afueras (casas grandes, vida residencial)
    "Carretera al Salvador": ZoneConfig(base_price_m2=130, weight=0.10, apt_share=0.35, center_to_business_km=9.0, typical_security=0.78, typical_amenities=0.55),

    # Intermedias / urbanas
    "Zona 11": ZoneConfig(base_price_m2=98,  weight=0.06, apt_share=0.60, center_to_business_km=3.5, typical_security=0.70, typical_amenities=0.45),
    "Zona 5":  ZoneConfig(base_price_m2=95,  weight=0.04, apt_share=0.60, center_to_business_km=3.5, typical_security=0.66, typical_amenities=0.42),
    "Zona 2":  ZoneConfig(base_price_m2=92,  weight=0.03, apt_share=0.60, center_to_business_km=3.2, typical_security=0.64, typical_amenities=0.40),
    "Zona 8":  ZoneConfig(base_price_m2=90,  weight=0.03, apt_share=0.55, center_to_business_km=4.2, typical_security=0.60, typical_amenities=0.38),
    "Zona 7":  ZoneConfig(base_price_m2=90,  weight=0.06, apt_share=0.50, center_to_business_km=4.5, typical_security=0.58, typical_amenities=0.35),
    "Zona 1":  ZoneConfig(base_price_m2=88,  weight=0.04, apt_share=0.60, center_to_business_km=3.0, typical_security=0.62, typical_amenities=0.40),
    "Zona 12": ZoneConfig(base_price_m2=88,  weight=0.03, apt_share=0.55, center_to_business_km=5.0, typical_security=0.58, typical_amenities=0.35),
    "Zona 6":  ZoneConfig(base_price_m2=82,  weight=0.04, apt_share=0.50, center_to_business_km=4.0, typical_security=0.56, typical_amenities=0.34),
    "Zona 17": ZoneConfig(base_price_m2=80,  weight=0.03, apt_share=0.45, center_to_business_km=5.5, typical_security=0.54, typical_amenities=0.32),
    "Zona 3":  ZoneConfig(base_price_m2=75,  weight=0.03, apt_share=0.55, center_to_business_km=4.0, typical_security=0.52, typical_amenities=0.31),
    "Zona 18": ZoneConfig(base_price_m2=70,  weight=0.03, apt_share=0.45, center_to_business_km=6.0, typical_security=0.50, typical_amenities=0.30),
}

ALL_ZONES = list(ZONES.keys())
ZONE_WEIGHTS = np.array([ZONES[z].weight for z in ALL_ZONES], dtype=float)
ZONE_WEIGHTS = ZONE_WEIGHTS / ZONE_WEIGHTS.sum()  # normalizar


def _draw_zone(rng: np.random.Generator) -> str:
    return rng.choice(ALL_ZONES, p=ZONE_WEIGHTS)


def _choose_property_type(zone: str, rng: np.random.Generator) -> str:
    apt_share = ZONES[zone].apt_share
    return "Apartamento" if rng.random() < apt_share else "Casa"


def _size_and_rooms(prop_type: str, zone: str, rng: np.random.Generator) -> Tuple[int, int, int]:
    """Tamaños coherentes por tipo/zonas (apartamentos más compactos en zonas verticales)."""
    vertical_zones = {"Zona 10", "Zona 14", "Zona 9", "Zona 4", "Zona 13", "Zona 15"}
    if prop_type == "Apartamento":
        base_mu = 72 if zone in vertical_zones else 85
        base_sigma = 18
        size = int(np.clip(rng.normal(base_mu, base_sigma), 35, 180))
    else:
        # Casas más grandes, especialmente en CES / 16 / 15
        base_mu = 160 if zone in {"Carretera al Salvador", "Zona 16", "Zona 15"} else 135
        size = int(np.clip(rng.normal(base_mu, 42), 70, 380))

    # Dormitorios y baños proporcionales al tamaño
    bedrooms = int(np.clip(round(size / (28 if prop_type == "Apartamento" else 35)), 1, 6))
    bathrooms = int(np.clip(round(max(1, bedrooms - 1 + rng.normal(0.2, 0.6))), 1, 5))
    return size, bedrooms, bathrooms


def _floors_and_age(prop_type: str, zone: str, rng: np.random.Generator) -> Tuple[int, int, int]:
    """Piso (si aplica) y antigüedad."""
    highrise_zones = {"Zona 10", "Zona 14", "Zona 9", "Zona 4", "Zona 13", "Zona 15"}
    newer_zones = {"Zona 10", "Zona 14", "Zona 15", "Zona 16", "Zona 4", "Zona 13"}

    if prop_type == "Apartamento":
        max_floor = 18 if zone in highrise_zones else 10
        floor = int(rng.integers(1, max_floor + 1))
        has_elevator = 1 if (floor >= 4 or zone in highrise_zones) else int(rng.random() < 0.5)
    else:
        floor, has_elevator = 0, 0

    base_age = 12 if zone in newer_zones else 18
    age_years = int(np.clip(rng.normal(base_age, 9), 0, 50))
    return floor, has_elevator, age_years


def _amenities(zone: str, rng: np.random.Generator) -> Dict[str, int]:
    conf = ZONES[zone]
    has_security = int(rng.random() < conf.typical_security)
    has_pool = int(rng.random() < conf.typical_amenities * 0.6)
    has_gym = int(rng.random() < conf.typical_amenities * 0.7)
    has_social = int(rng.random() < conf.typical_amenities * 0.8)
    furnished = int(rng.random() < (0.25 if zone in {"Zona 10", "Zona 14"} else 0.12))
    balcony = int(rng.random() < 0.55)
    garden = int(rng.random() < (0.15 if zone in {"Zona 10", "Zona 14", "Zona 9"} else 0.35))
    parking = int(np.clip(int(rng.normal(1.4, 0.7)), 0, 3))
    pet_friendly = int(rng.random() < 0.55)
    return dict(
        has_security=has_security, has_pool=has_pool, has_gym=has_gym, has_social=has_social,
        furnished=furnished, balcony=balcony, garden=garden, parking_spaces=parking, pet_friendly=pet_friendly
    )


def _environment(zone: str, rng: np.random.Generator) -> Tuple[float, float, int]:
    """Distancias y entorno (ruido, vista)."""
    # Distancia a centros de negocio y a transporte (en km)
    center_km = max(0.2, rng.normal(ZONES[zone].center_to_business_km, 0.8))
    transit_km = max(0.2, rng.normal(1.2 if zone in {"Zona 10", "Zona 14", "Zona 4", "Zona 9"} else 2.2, 0.9))

    # Vista: mejor en 15/16/CES (y algo en 14 por alturas)
    if zone in {"Zona 15", "Zona 16", "Carretera al Salvador", "Zona 14"}:
        base_view = 3.2
    else:
        base_view = 2.6
    view_quality = int(np.clip(round(base_view + rng.normal(0.0, 0.9)), 1, 5))

    # Ruido: mayor en zonas céntricas y vías principales (incluye Zona 1)
    base_noise = 3.4 if zone in {"Zona 10", "Zona 9", "Zona 4", "Zona 13", "Zona 1"} else 2.6
    noise_level = int(np.clip(round(base_noise + rng.normal(0.0, 0.9)), 1, 5))
    return center_km, transit_km, view_quality - noise_level  # score de “atractivo exterior”


def _price_formula(row: pd.Series, rng: np.random.Generator) -> float:
    """
    Calcula el precio en GTQ de forma explicable.
    1) Base por m² de la zona
    2) Ajustes multiplicativos por tipo/tamaño/calidad
    3) Ajustes aditivos (estacionamientos, amenities)
    """
    base_m2 = ZONES[row["zone"]].base_price_m2

    # Multiplicadores
    m_type = 1.00 if row["property_type"] == "Apartamento" else 0.95
    m_age = 1.0 - min(row["age_years"], 40) * 0.006
    m_view = 1.0 + (row["view_quality"] - 3) * 0.03
    m_noise = 1.0 - (row["noise_level"] - 3) * 0.025

    # Premium por piso alto si hay elevador
    if row["property_type"] == "Apartamento" and row["has_elevator"] == 1 and row["floor"] >= 8:
        m_floor = 1.06
    elif row["property_type"] == "Apartamento" and row["floor"] >= 4:
        m_floor = 1.03
    else:
        m_floor = 1.00

    # Penalización por distancias (negativas)
    m_center = 1.0 - 0.02 * math.log1p(row["distance_to_business_km"])
    m_transit = 1.0 - 0.015 * math.log1p(row["distance_to_transit_km"])

    # Composición de multiplicadores (acotamos)
    mult = np.clip(m_type * m_age * m_view * m_noise * m_floor * m_center * m_transit, 0.6, 1.35)

    # Aditivos (GTQ)
    add_amenities = (
        row["parking_spaces"] * 175
        + row["has_security"] * 250
        + row["has_pool"] * 220
        + row["has_gym"] * 150
        + row["has_social"] * 120
        + row["furnished"] * 350
        + row["balcony"] * 90
        + row["garden"] * 180
        + (row["pet_friendly"] * 60)
    )

    # Precio base
    price = row["size_m2"] * base_m2 * mult + add_amenities

    # Ajuste por dormitorios/baños vs metraje esperado
    expected_beds = max(1, round(row["size_m2"] / (28 if row["property_type"] == "Apartamento" else 35)))
    price *= 1.0 + 0.01 * (row["bedrooms"] - expected_beds)
    price *= 1.0 + 0.008 * (row["bathrooms"] - max(1, expected_beds - 1))

    # Ruido estocástico (mercado/negociación)
    price += rng.normal(0, 250)

    return float(max(price, 1800))


def generate_rent_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        zone = _draw_zone(rng)
        prop_type = _choose_property_type(zone, rng)
        size_m2, bedrooms, bathrooms = _size_and_rooms(prop_type, zone, rng)
        floor, has_elevator, age_years = _floors_and_age(prop_type, zone, rng)
        amen = _amenities(zone, rng)
        center_km, transit_km, ext_score = _environment(zone, rng)

        # Derivar ruido/vista a partir del score exterior (+ variabilidad)
        noise_level = np.clip(3 + (-ext_score) + rng.normal(0, 0.6), 1, 5)
        view_quality = np.clip(3 + ext_score + rng.normal(0, 0.6), 1, 5)

        row = dict(
            listing_id=i,
            zone=zone,
            property_type=prop_type,
            size_m2=int(size_m2),
            bedrooms=int(bedrooms),
            bathrooms=int(bathrooms),
            age_years=int(age_years),
            floor=int(floor),
            has_elevator=int(has_elevator),
            distance_to_business_km=round(center_km, 2),
            distance_to_transit_km=round(transit_km, 2),
            view_quality=int(round(view_quality)),
            noise_level=int(round(noise_level)),
            **amen
        )
        price = _price_formula(pd.Series(row), rng)
        row["rent_price_gtq"] = round(price, 2)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Derivados útiles
    df["price_per_m2"] = (df["rent_price_gtq"] / df["size_m2"]).round(2)
    df["age_bucket"] = pd.cut(df["age_years"], bins=[-1, 5, 15, 30, 100], labels=["0-5", "6-15", "16-30", "30+"])
    df["is_premium_zone"] = df["zone"].isin(["Zona 10", "Zona 14", "Zona 15", "Zona 16", "Carretera al Salvador"]).astype(int)

    return df


def save_dataset(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="Número de registros a generar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--out", type=str, default="data/raw/rent_guatemala.csv", help="Ruta de salida del CSV")
    args = parser.parse_args()

    df = generate_rent_data(n=args.n, seed=args.seed)
    save_dataset(df, args.out)

    # Resumen breve en consola
    print("✅ Dataset generado:", args.out)
    print("Filas:", len(df))
    print("Columnas:", list(df.columns))
    print("\nTop 8 zonas por precio promedio (GTQ):")
    print(df.groupby("zone")["rent_price_gtq"].mean().sort_values(ascending=False).round(0).head(8))
    print("\nCorrelaciones con el precio (top 10 absolutas):")
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr(numeric_only=True)["rent_price_gtq"].sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr.head(10).round(3))


if __name__ == "__main__":
    main()
