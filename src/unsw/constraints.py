from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.unsw.data import load_official_data

FEATURE_ORDER = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

OFFICIAL_TRAIN_DF, _ = load_official_data()

CATEGORICAL_VALUE_MAP = {
    "proto": sorted(OFFICIAL_TRAIN_DF["proto"].dropna().unique().tolist()),
    "service": sorted(OFFICIAL_TRAIN_DF["service"].dropna().unique().tolist()),
    "state": sorted(OFFICIAL_TRAIN_DF["state"].dropna().unique().tolist()),
}

NUMERIC_MIN_MAX = {
    feature: (float(OFFICIAL_TRAIN_DF[feature].min()), float(OFFICIAL_TRAIN_DF[feature].max()))
    for feature in FEATURE_ORDER
    if pd.api.types.is_numeric_dtype(OFFICIAL_TRAIN_DF[feature])
}

BINARY_FEATURES = {"is_ftp_login", "is_sm_ips_ports"}
INTEGER_FEATURES = {
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
}

NON_NEGATIVE_FEATURES = set(NUMERIC_MIN_MAX)

PRIMARY_CATEGORY_COUNTS = {
    "DIRECTLY_CONTROLLABLE": 0,
    "DERIVED_DEPENDENT": 0,
    "NON_CONTROLLABLE": 0,
    "ATTACK_REQUIRED_PRESERVED": 0,
}

FEATURE_METADATA = [
    {
        "feature": "dur",
        "data_type": "numeric",
        "semantic_description": "Record total duration.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Adjust flow duration within observed bounds while keeping the flow behavior plausible.",
        "valid_min_observed": NUMERIC_MIN_MAX["dur"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dur"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["sinpkt", "dinpkt", "sjit", "djit", "rate"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. Duration is a traffic-behavior parameter and can vary without necessarily invalidating the attack semantics.",
        "confidence_level": "HIGH",
        "justification": "This is a timing feature that can plausibly vary with attacker-controlled traffic behavior and is not a direct identifier or label proxy.",
    },
    {
        "feature": "proto",
        "data_type": "categorical",
        "semantic_description": "Transaction protocol.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Choose a valid protocol category consistent with the intended traffic behavior.",
        "valid_min_observed": "N/A",
        "valid_max_observed": "N/A",
        "integer_required": False,
        "nonnegative_required": False,
        "related_features": ["service", "state", "swin", "dwin", "stcpb", "dtcpb"],
        "dependency_rule": "Protocol may conditionally affect valid service/state combinations; exact dependency requires domain verification.",
        "attack_semantics_requirement": "Protocol is part of the traffic semantics; changing it should not erase the attack intent unless the protocol choice is an invalid attack mode.",
        "confidence_level": "HIGH",
        "justification": "The dataset explicitly lists protocol categories, and these are labels rather than derived numeric counters.",
    },
    {
        "feature": "service",
        "data_type": "categorical",
        "semantic_description": "Application/service indicator such as http, ftp, dns, ssh, and '-' if not used.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Assign a valid service label consistent with the traffic scenario and protocol.",
        "valid_min_observed": "N/A",
        "valid_max_observed": "N/A",
        "integer_required": False,
        "nonnegative_required": False,
        "related_features": ["proto", "state", "trans_depth", "response_body_len", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"],
        "dependency_rule": "Service-specific behavior is often conditional; exact cross-feature validity requires protocol/service-specific review.",
        "attack_semantics_requirement": "A malicious flow may retain its attack semantics even when the application service label varies within valid categories, but a service change must remain context-consistent.",
        "confidence_level": "HIGH",
        "justification": "The documentation identifies service as a categorical traffic descriptor and it is not a derived feature.",
    },
    {
        "feature": "state",
        "data_type": "categorical",
        "semantic_description": "Connection state, such as CON, ECO, FIN, INT, REQ, RST, URN, or no.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Choose a valid state category that matches the connection behavior.",
        "valid_min_observed": "N/A",
        "valid_max_observed": "N/A",
        "integer_required": False,
        "nonnegative_required": False,
        "related_features": ["proto", "service", "ct_state_ttl"],
        "dependency_rule": "State is conditionally related to protocol and connection setup; exact logic requires protocol/domain verification.",
        "attack_semantics_requirement": "State may be part of attack semantics, but it is not a label itself and should be treated as a valid traffic state rather than a direct identifier.",
        "confidence_level": "HIGH",
        "justification": "The feature description explicitly names state categories and their protocol dependence, which supports a categorical validity rule rather than arbitrary mutation.",
    },
    {
        "feature": "spkts",
        "data_type": "integer",
        "semantic_description": "Source to destination packet count.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Increase or decrease packet count while preserving nonnegative integer values and realistic traffic volumes.",
        "valid_min_observed": NUMERIC_MIN_MAX["spkts"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["spkts"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dpkts", "sbytes", "dbytes", "sload", "dload", "smean"],
        "dependency_rule": "Packet counts are correlated with bytes and mean packet size; exact reconstruction rule requires verification.",
        "attack_semantics_requirement": "No; packet counts can vary as part of the attack behavior without invalidating the flow as attack traffic.",
        "confidence_level": "HIGH",
        "justification": "This is a standard packet-count variable with clear integer semantics and no indication that it is a latent or derived identifier.",
    },
    {
        "feature": "dpkts",
        "data_type": "integer",
        "semantic_description": "Destination to source packet count.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Adjust destination packet volume within nonnegative integer bounds.",
        "valid_min_observed": NUMERIC_MIN_MAX["dpkts"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dpkts"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["spkts", "dbytes", "sbytes", "dload", "dmean"],
        "dependency_rule": "Packet counts are linked to bytes and mean packet size; exact formula requires verification.",
        "attack_semantics_requirement": "No; this is a traffic-volume feature rather than a direct label or identity field.",
        "confidence_level": "HIGH",
        "justification": "This is plainly a count feature with standard integer semantics and no evidence of hidden label leakage.",
    },
    {
        "feature": "sbytes",
        "data_type": "integer",
        "semantic_description": "Source to destination transaction bytes.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Adjust source-byte volume while keeping it nonnegative and consistent with packet count and mean size.",
        "valid_min_observed": NUMERIC_MIN_MAX["sbytes"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["sbytes"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dbytes", "spkts", "sload", "smean", "sloss"],
        "dependency_rule": "Strong empirical correlation with sloss and packet count is present; exact mathematical dependency requires verification.",
        "attack_semantics_requirement": "No. This is a byte-volume statistic and not an ID-like field.",
        "confidence_level": "HIGH",
        "justification": "The dataset description names this as a transaction byte count, which is a standard directly controllable traffic measure under the assumption of plausible flow behavior.",
    },
    {
        "feature": "dbytes",
        "data_type": "integer",
        "semantic_description": "Destination to source transaction bytes.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Adjust destination-byte volume while preserving nonnegative integer count semantics.",
        "valid_min_observed": NUMERIC_MIN_MAX["dbytes"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dbytes"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["sbytes", "dpkts", "dload", "dmean", "dloss"],
        "dependency_rule": "Empirical correlation with dloss and dpkts is strong; exact dependence requires verification.",
        "attack_semantics_requirement": "No. This is a byte-volume feature rather than a direct attack marker.",
        "confidence_level": "HIGH",
        "justification": "This is a standard byte counter attached to a flow and can be changed plausibly by varying traffic behavior.",
    },
    {
        "feature": "rate",
        "data_type": "numeric",
        "semantic_description": "Traffic rate, likely bits per second or a rate-based summary.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Only modify indirectly by changing bytes and duration; direct independent mutation is not recommended.",
        "valid_min_observed": NUMERIC_MIN_MAX["rate"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["rate"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["dur", "sbytes", "dbytes", "sload", "dload"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. This is a rate summary and should remain consistent with traffic volume and duration if modified indirectly.",
        "confidence_level": "MEDIUM",
        "justification": "Rate is usually derived from bytes and duration, but the exact dataset-specific formula is not explicit in the current project’s documentation.",
    },
    {
        "feature": "sttl",
        "data_type": "integer",
        "semantic_description": "Source to destination time-to-live value.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust within a valid TTL range if the flow semantics permit it.",
        "valid_min_observed": NUMERIC_MIN_MAX["sttl"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["sttl"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dttl", "ct_state_ttl"],
        "dependency_rule": "TTL values are semantically related to state and routing behavior; exact formula not required to define a valid range.",
        "attack_semantics_requirement": "No. TTL is a transport-layer attribute and not by itself a direct label or identity field.",
        "confidence_level": "HIGH",
        "justification": "The project documentation defines TTL as a field with standard nonnegative integer semantics.",
    },
    {
        "feature": "dttl",
        "data_type": "integer",
        "semantic_description": "Destination to source time-to-live value.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust within realistic TTL bounds if the traffic scenario permits it.",
        "valid_min_observed": NUMERIC_MIN_MAX["dttl"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dttl"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["sttl", "ct_state_ttl"],
        "dependency_rule": "TTL values are semantically linked to protocol and routing behavior; exact formula requires domain verification.",
        "attack_semantics_requirement": "No. This is an environmental/transport feature rather than a label or identity feature.",
        "confidence_level": "HIGH",
        "justification": "TTL fields are explicitly defined as nonnegative integer counters and are not directly derived from labels.",
    },
    {
        "feature": "sload",
        "data_type": "numeric",
        "semantic_description": "Source bits per second.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Only modify indirectly by changing traffic volume and duration; do not edit arbitrarily in isolation.",
        "valid_min_observed": NUMERIC_MIN_MAX["sload"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["sload"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["sbytes", "spkts", "dur", "rate"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. This is a rate summary that should stay consistent with traffic behavior rather than being manipulated independently.",
        "confidence_level": "MEDIUM",
        "justification": "The name indicates a rate-type feature, and rate-type summaries are typically derived from other flow quantities rather than independently controllable.",
    },
    {
        "feature": "dload",
        "data_type": "numeric",
        "semantic_description": "Destination bits per second.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Adjust only through consistent changes to byte volume and duration.",
        "valid_min_observed": NUMERIC_MIN_MAX["dload"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dload"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["dbytes", "dpkts", "dur", "rate"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. This is a derived traffic-rate metric and should not be changed independently.",
        "confidence_level": "MEDIUM",
        "justification": "This appears to be a derived traffic-rate measure, which makes direct independent mutation unsafe under a realistic constraint model.",
    },
    {
        "feature": "sloss",
        "data_type": "integer",
        "semantic_description": "Source packets retransmitted or dropped.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Change only within nonnegative integer bounds consistent with packet loss behavior.",
        "valid_min_observed": NUMERIC_MIN_MAX["sloss"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["sloss"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dloss", "spkts", "sbytes", "smean"],
        "dependency_rule": "Strong empirical correlation with packet and byte counts is present; exact reconstruction rule requires verification.",
        "attack_semantics_requirement": "No. Packet loss counts are traffic-behavior features, not direct identifiers.",
        "confidence_level": "HIGH",
        "justification": "The feature definition clearly states that it records dropped or retransmitted packets, which is a controllable traffic characteristic rather than a latent identity field.",
    },
    {
        "feature": "dloss",
        "data_type": "integer",
        "semantic_description": "Destination packets retransmitted or dropped.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust within realistic loss counts while keeping the flow consistent.",
        "valid_min_observed": NUMERIC_MIN_MAX["dloss"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dloss"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["sloss", "dpkts", "dbytes", "dmean"],
        "dependency_rule": "Strong empirical correlation with dpkts and dbytes is present; exact formula requires verification.",
        "attack_semantics_requirement": "No. This is a traffic-quality metric, not a label or identity field.",
        "confidence_level": "HIGH",
        "justification": "The description clearly labels this as a packet loss count, which is an ordinary traffic-feature quantity.",
    },
    {
        "feature": "sinpkt",
        "data_type": "numeric",
        "semantic_description": "Source inter-packet arrival time.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Adjust packet timing within nonnegative values consistent with the traffic pattern.",
        "valid_min_observed": NUMERIC_MIN_MAX["sinpkt"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["sinpkt"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["dinpkt", "sjit", "djit", "dur", "spkts"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. Timing behavior is part of the flow and can be plausibly altered without changing the attack label.",
        "confidence_level": "HIGH",
        "justification": "Inter-packet timing is a direct behavior metric and not a class label or identifier.",
    },
    {
        "feature": "dinpkt",
        "data_type": "numeric",
        "semantic_description": "Destination inter-packet arrival time.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Adjust destination timing values while keeping them nonnegative and realistic.",
        "valid_min_observed": NUMERIC_MIN_MAX["dinpkt"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dinpkt"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["sinpkt", "sjit", "djit", "dur", "dpkts"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. This is a timing feature and not a direct attack label or identity field.",
        "confidence_level": "HIGH",
        "justification": "The dataset description defines it as an inter-packet timing value, which is a standard traffic-behavior feature.",
    },
    {
        "feature": "sjit",
        "data_type": "numeric",
        "semantic_description": "Source jitter.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Modify timing jitter within nonnegative observed bounds.",
        "valid_min_observed": NUMERIC_MIN_MAX["sjit"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["sjit"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["djit", "sinpkt", "dinpkt", "dur"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. Jitter is traffic behavior and can vary independently of attack semantics within realistic constraints.",
        "confidence_level": "HIGH",
        "justification": "The documentation identifies this as jitter, which is an ordinary timing summary rather than an identifier-like feature.",
    },
    {
        "feature": "djit",
        "data_type": "numeric",
        "semantic_description": "Destination jitter.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "high",
        "allowed_operation": "Modify destination-side jitter within realistic ranges.",
        "valid_min_observed": NUMERIC_MIN_MAX["djit"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["djit"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["sjit", "sinpkt", "dinpkt", "dur"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. This is traffic timing behavior, not a label or identity field.",
        "confidence_level": "HIGH",
        "justification": "Jitter is clearly a network timing metric and a plausible attacker-controlled property in flow reconstruction.",
    },
    {
        "feature": "swin",
        "data_type": "integer",
        "semantic_description": "Source TCP window advertisement value.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust only within valid TCP window values and keep the protocol semantics consistent.",
        "valid_min_observed": NUMERIC_MIN_MAX["swin"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["swin"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dwin", "stcpb", "dtcpb", "proto"],
        "dependency_rule": "TCP window fields are protocol-dependent and correlated with destination window, exact formula not required for validity.",
        "attack_semantics_requirement": "No. This configuration field can vary with traffic behavior but is not a direct label or target identifier.",
        "confidence_level": "HIGH",
        "justification": "The feature is documented as a TCP window value, which is a transport-layer setting rather than a latent dataset artifact.",
    },
    {
        "feature": "stcpb",
        "data_type": "integer",
        "semantic_description": "Source TCP base sequence number.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust only within valid nonnegative TCP sequence ranges and keep protocol semantics consistent.",
        "valid_min_observed": NUMERIC_MIN_MAX["stcpb"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["stcpb"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dtcpb", "swin", "dwin", "proto"],
        "dependency_rule": "TCP sequence numbers are protocol-specific and can be correlated with other TCP window/sequence fields; exact relation requires verification.",
        "attack_semantics_requirement": "No. Sequence numbers are traffic-setup metadata, not a direct ID field.",
        "confidence_level": "HIGH",
        "justification": "The feature definition explicitly names TCP base sequence numbers and therefore indicates a transport-layer configuration field.",
    },
    {
        "feature": "dtcpb",
        "data_type": "integer",
        "semantic_description": "Destination TCP base sequence number.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust only within valid TCP sequence ranges compatible with the connection.",
        "valid_min_observed": NUMERIC_MIN_MAX["dtcpb"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dtcpb"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["stcpb", "swin", "dwin", "proto"],
        "dependency_rule": "TCP sequence numbers are protocol-specific and related to source sequence numbers; exact relation requires verification.",
        "attack_semantics_requirement": "No. This is TCP connection metadata rather than a label or identity feature.",
        "confidence_level": "HIGH",
        "justification": "The feature description names these as TCP base sequence numbers, supporting direct controllability within transport semantics.",
    },
    {
        "feature": "dwin",
        "data_type": "integer",
        "semantic_description": "Destination TCP window advertisement value.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust only within valid TCP window values and consistent destination-side transport semantics.",
        "valid_min_observed": NUMERIC_MIN_MAX["dwin"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dwin"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["swin", "stcpb", "dtcpb", "proto"],
        "dependency_rule": "TCP window fields are protocol-dependent; exact algebraic relation not specified in the current documentation.",
        "attack_semantics_requirement": "No. This is transport metadata and should not be treated as direct label leakage.",
        "confidence_level": "HIGH",
        "justification": "The feature is explicitly described as a TCP window advertisement value, which is transport configuration rather than a label proxy.",
    },
    {
        "feature": "tcprtt",
        "data_type": "numeric",
        "semantic_description": "TCP connection setup round-trip time, the sum of synack and ackdat.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Only modify indirectly through the underlying TCP handshake timing components.",
        "valid_min_observed": NUMERIC_MIN_MAX["tcprtt"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["tcprtt"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["synack", "ackdat"],
        "dependency_rule": "tcprtt = synack + ackdat (documented in feature description).",
        "attack_semantics_requirement": "No. If the TCP handshake semantics are preserved, the derived RTT value should remain consistent with synack and ackdat.",
        "confidence_level": "HIGH",
        "justification": "The dataset documentation explicitly states that the RTT is the sum of synack and ackdat, so this is a verified dependent feature.",
    },
    {
        "feature": "synack",
        "data_type": "numeric",
        "semantic_description": "TCP connection setup time between SYN and SYN_ACK.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust TCP handshake timing within realistic nonnegative timing values.",
        "valid_min_observed": NUMERIC_MIN_MAX["synack"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["synack"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["ackdat", "tcprtt"],
        "dependency_rule": "synack and ackdat together define tcprtt; exact arithmetic relation is documented.",
        "attack_semantics_requirement": "No. This is connection timing, not a direct label field.",
        "confidence_level": "HIGH",
        "justification": "The documentation defines this as a standard TCP setup timing metric and includes its role in RTT composition.",
    },
    {
        "feature": "ackdat",
        "data_type": "numeric",
        "semantic_description": "TCP connection setup time between SYN_ACK and ACK packets.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust handshake timing while preserving nonnegative, domain-valid timing values.",
        "valid_min_observed": NUMERIC_MIN_MAX["ackdat"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ackdat"][1],
        "integer_required": False,
        "nonnegative_required": True,
        "related_features": ["synack", "tcprtt"],
        "dependency_rule": "ackdat + synack = tcprtt (documented in feature description).",
        "attack_semantics_requirement": "No. This is transport timing metadata and not a direct label or identity field.",
        "confidence_level": "HIGH",
        "justification": "It is explicitly stated that RTT is the sum of synack and ackdat, so the dependency is verified in the provided documentation.",
    },
    {
        "feature": "smean",
        "data_type": "integer",
        "semantic_description": "Mean packet size transmitted by the source.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Only modify indirectly by changing packet size and byte/packet relationships. Do not change independently if the flow must remain internally consistent.",
        "valid_min_observed": NUMERIC_MIN_MAX["smean"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["smean"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["dmean", "sbytes", "dbytes", "spkts", "dpkts"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. Mean packet size is a derived traffic statistic and should be internally consistent with bytes and packets.",
        "confidence_level": "MEDIUM",
        "justification": "The description indicates a mean packet size, which is typically computed from bytes and packet counts rather than directly set as an independent traffic property.",
    },
    {
        "feature": "dmean",
        "data_type": "integer",
        "semantic_description": "Mean packet size transmitted by the destination.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Modify via related byte/packet changes only; avoid independent mutation.",
        "valid_min_observed": NUMERIC_MIN_MAX["dmean"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["dmean"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["smean", "dbytes", "sbytes", "dpkts", "spkts"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "No. Mean packet size is a summarizing statistic and should remain consistent with traffic volume.",
        "confidence_level": "MEDIUM",
        "justification": "The project documentation identifies this as a mean packet size feature, which suggests computation from other traffic quantities instead of direct independent setting.",
    },
    {
        "feature": "trans_depth",
        "data_type": "integer",
        "semantic_description": "Pipelined depth into HTTP request/response transaction.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust only when the service context permits HTTP transaction depth changes.",
        "valid_min_observed": NUMERIC_MIN_MAX["trans_depth"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["trans_depth"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "response_body_len", "ct_flw_http_mthd"],
        "dependency_rule": "HTTP-specific; exact logic depends on how request/response transactions are represented in the flow.",
        "attack_semantics_requirement": "If the traffic remains an HTTP-based attack, transaction depth should remain compatible with that service context.",
        "confidence_level": "MEDIUM",
        "justification": "The feature description explicitly ties this to HTTP transaction depth, so the value is service-dependent and should not be treated as universally independent.",
    },
    {
        "feature": "response_body_len",
        "data_type": "integer",
        "semantic_description": "Actual uncompressed content size of server HTTP response body.",
        "constraint_category": "DIRECTLY_CONTROLLABLE",
        "attacker_controllability": "medium",
        "allowed_operation": "Adjust only in HTTP-like traffic and keep the content length consistent with the response.",
        "valid_min_observed": NUMERIC_MIN_MAX["response_body_len"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["response_body_len"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "trans_depth", "ct_flw_http_mthd"],
        "dependency_rule": "HTTP-specific; exact relation to service and transaction depth requires verification.",
        "attack_semantics_requirement": "If the flow is HTTP-based, response body length should remain consistent with the attack semantics; otherwise it may be irrelevant.",
        "confidence_level": "MEDIUM",
        "justification": "This is clearly HTTP-response content size and therefore not a universally independent field; it is service-conditioned.",
    },
    {
        "feature": "ct_srv_src",
        "data_type": "integer",
        "semantic_description": "Number of connections with the same service and source address in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Do not set independently; treat as a context-dependent count frozen in constrained attacks.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_srv_src"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_srv_src"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "ct_srv_dst", "ct_dst_src_ltm", "ct_src_ltm"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is contextual network-history data and should be frozen or recomputed from valid surrounding traffic context rather than changed in isolation.",
        "confidence_level": "HIGH",
        "justification": "The feature description explicitly names it as a connection-history count derived from surrounding traffic context, so it should be treated as dependent and not directly manipulated.",
    },
    {
        "feature": "ct_state_ttl",
        "data_type": "integer",
        "semantic_description": "Number of connections for each state according to source/destination TTL ranges.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Freeze or recompute from surrounding connection history; do not mutate independently.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_state_ttl"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_state_ttl"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["state", "sttl", "dttl"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is a contextual count summarizing state/TTL relationships; it should be preserved when the attack semantics are maintained.",
        "confidence_level": "HIGH",
        "justification": "The documentation explicitly describes it as a count over state and TTL ranges, which is a contextual summary and not a directly modifiable physical attribute.",
    },
    {
        "feature": "ct_dst_ltm",
        "data_type": "integer",
        "semantic_description": "Number of connections to the same destination address in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Treat as a contextual count; do not mutate independently.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_dst_ltm"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_dst_ltm"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_srv_dst"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is a surrounding-traffic count and should be preserved as context unless recomputed from a valid traffic history.",
        "confidence_level": "HIGH",
        "justification": "The descriptor clearly names it as a connection-history count, which makes independent mutation unrealistic within a realistic attack model.",
    },
    {
        "feature": "ct_src_dport_ltm",
        "data_type": "integer",
        "semantic_description": "Number of flows from the same source and destination port in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Freeze or recompute from surrounding connection history; avoid direct mutation.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_src_dport_ltm"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_src_dport_ltm"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["ct_src_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This feature depends on historical connection patterns and should remain frozen unless the surrounding traffic model is explicitly reconstructed.",
        "confidence_level": "HIGH",
        "justification": "Its description is explicitly connection-history based, making it a contextual derived feature rather than a directly set traffic property.",
    },
    {
        "feature": "ct_dst_sport_ltm",
        "data_type": "integer",
        "semantic_description": "Number of flows with the same destination and source port in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Treat as background traffic context; frozen in constrained attacks.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_dst_sport_ltm"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_dst_sport_ltm"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["ct_src_dport_ltm", "ct_dst_ltm", "ct_dst_src_ltm"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is contextual traffic-history data and should not be independently mutated in a realistic attack model.",
        "confidence_level": "HIGH",
        "justification": "The feature definition ties it to historical connection patterns, which makes it context-derived rather than directly controllable.",
    },
    {
        "feature": "ct_dst_src_ltm",
        "data_type": "integer",
        "semantic_description": "Number of connections with the same source and destination address in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Freeze or recompute based on surrounding traffic history; do not vary independently.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_dst_src_ltm"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_dst_src_ltm"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is a surrounding-traffic count, so its mutation should be constrained to realistic environmental history changes only.",
        "confidence_level": "HIGH",
        "justification": "The description names historical same-source/destination counts, which makes it a contextual count and not a feature that can be set directly.",
    },
    {
        "feature": "is_ftp_login",
        "data_type": "binary",
        "semantic_description": "Indicator for whether the FTP session was accessed by user and password.",
        "constraint_category": "ATTACK_REQUIRED_PRESERVED",
        "attacker_controllability": "low",
        "allowed_operation": "Keep consistent with the FTP session semantics; do not change unless the FTP login semantics of the flow are explicitly being redesigned.",
        "valid_min_observed": NUMERIC_MIN_MAX["is_ftp_login"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["is_ftp_login"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "ct_ftp_cmd"],
        "dependency_rule": "Only meaningful for FTP activity; exact condition depends on service semantics and flow context.",
        "attack_semantics_requirement": "This can be part of the malicious FTP behavior; it should be preserved when the attack flow is intended to remain an FTP login-based attack.",
        "confidence_level": "MEDIUM",
        "justification": "The feature is semantically meaningful only in FTP contexts and is near-constant in the data, which makes it a high-risk feature to mutate without verifying login semantics.",
    },
    {
        "feature": "ct_ftp_cmd",
        "data_type": "integer",
        "semantic_description": "Number of FTP commands in the session.",
        "constraint_category": "ATTACK_REQUIRED_PRESERVED",
        "attacker_controllability": "low",
        "allowed_operation": "It should be preserved unless the FTP behavior is explicitly redesigned; otherwise treat as a frozen contextual session property.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_ftp_cmd"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_ftp_cmd"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "is_ftp_login"],
        "dependency_rule": "Only meaningful under FTP service; exact dependency requires verification.",
        "attack_semantics_requirement": "This is directly tied to FTP session usage and should be preserved when the malicious session semantics are kept intact.",
        "confidence_level": "MEDIUM",
        "justification": "The feature is highly concentrated and highly correlated with is_ftp_login, suggesting it encodes specific FTP action patterns rather than arbitrary traffic variation.",
    },
    {
        "feature": "ct_flw_http_mthd",
        "data_type": "integer",
        "semantic_description": "Number of HTTP flows with methods such as Get and Post.",
        "constraint_category": "ATTACK_REQUIRED_PRESERVED",
        "attacker_controllability": "low",
        "allowed_operation": "Only modify when the attack-specific HTTP session behavior is explicitly being retained; otherwise freeze it.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_flw_http_mthd"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_flw_http_mthd"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "trans_depth", "response_body_len"],
        "dependency_rule": "HTTP-specific; exact relation to service and HTTP transactions requires verification.",
        "attack_semantics_requirement": "This is tied to HTTP method usage and should be treated as part of the attack semantics when the traffic is HTTP-based.",
        "confidence_level": "MEDIUM",
        "justification": "The descriptor explicitly states that it is an HTTP-method count, which makes it service-conditioned and not safely independent.",
    },
    {
        "feature": "ct_src_ltm",
        "data_type": "integer",
        "semantic_description": "Number of connections from the same source address in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Treat as a contextual historical count and freeze it in constrained attacks.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_src_ltm"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_src_ltm"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["ct_srv_src", "ct_dst_ltm", "ct_dst_src_ltm"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is a surrounding-traffic count; it should not be independently changed in a realistic attack model.",
        "confidence_level": "HIGH",
        "justification": "The feature description explicitly names it as a historical source-address connection count, which makes it a dependent contextual feature.",
    },
    {
        "feature": "ct_srv_dst",
        "data_type": "integer",
        "semantic_description": "Number of connections with the same service and destination address in recent history.",
        "constraint_category": "DERIVED_DEPENDENT",
        "attacker_controllability": "low",
        "allowed_operation": "Do not modify independently; freeze it or recompute from surrounding traffic history.",
        "valid_min_observed": NUMERIC_MIN_MAX["ct_srv_dst"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["ct_srv_dst"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["service", "ct_srv_src", "ct_dst_src_ltm", "ct_src_ltm"],
        "dependency_rule": "dependency exists; exact reconstruction rule requires verification",
        "attack_semantics_requirement": "This is a context-based count associated with surrounding traffic and should remain consistent with the flow history.",
        "confidence_level": "HIGH",
        "justification": "The feature is explicitly described as a count derived from a service/destination history window, so direct independent mutation is not realistic.",
    },
    {
        "feature": "is_sm_ips_ports",
        "data_type": "binary",
        "semantic_description": "Indicator that source and destination IP addresses and ports are the same.",
        "constraint_category": "NON_CONTROLLABLE",
        "attacker_controllability": "none",
        "allowed_operation": "Do not change directly in the canonical 42-feature model because it depends on address and port identity fields outside the model input.",
        "valid_min_observed": NUMERIC_MIN_MAX["is_sm_ips_ports"][0],
        "valid_max_observed": NUMERIC_MIN_MAX["is_sm_ips_ports"][1],
        "integer_required": True,
        "nonnegative_required": True,
        "related_features": ["proto", "service", "state"],
        "dependency_rule": "If the value is 1, the source/destination IP and source/destination port must be equal; exact evaluation requires address/port fields not present in the canonical predictor set.",
        "attack_semantics_requirement": "This should be treated as an identity/property indicator rather than a directly attacker-controlled traffic parameter.",
        "confidence_level": "HIGH",
        "justification": "The output description says this indicator is computed from source and destination IP/port equality, which is a contextual identity property outside the current canonical predictor set.",
    },
]

for row in FEATURE_METADATA:
    category = row["constraint_category"]
    if category in PRIMARY_CATEGORY_COUNTS:
        PRIMARY_CATEGORY_COUNTS[category] += 1


def _coerce_numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _categorical_valid(feature: str, value: Any) -> bool:
    allowed = CATEGORICAL_VALUE_MAP.get(feature)
    return allowed is not None and value in allowed


def build_feature_table_frame() -> pd.DataFrame:
    df = pd.DataFrame(FEATURE_METADATA)
    df = df[[
        "feature",
        "data_type",
        "semantic_description",
        "constraint_category",
        "attacker_controllability",
        "allowed_operation",
        "valid_min_observed",
        "valid_max_observed",
        "integer_required",
        "nonnegative_required",
        "related_features",
        "dependency_rule",
        "attack_semantics_requirement",
        "confidence_level",
        "justification",
    ]]
    return df


def _freeze_violation(feature: str, original: Mapping[str, Any], candidate: Mapping[str, Any]) -> str | None:
    metadata = {row["feature"]: row for row in FEATURE_METADATA}[feature]
    category = metadata["constraint_category"]
    if category in {"NON_CONTROLLABLE", "ATTACK_REQUIRED_PRESERVED"}:
        if feature in candidate and feature in original and candidate.get(feature) != original.get(feature):
            return f"frozen feature changed: {feature}"
    return None


def validate_candidate(original_sample: Mapping[str, Any], candidate_sample: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    warnings: list[str] = []
    changed_features: list[str] = []
    frozen_features_changed: list[str] = []

    for feature in FEATURE_ORDER:
        if feature not in candidate_sample:
            violations.append(f"missing feature in candidate: {feature}")
            continue

        if feature in original_sample and candidate_sample.get(feature) != original_sample.get(feature):
            changed_features.append(feature)

        metadata = {row["feature"]: row for row in FEATURE_METADATA}[feature]
        category = metadata["constraint_category"]
        candidate_value = candidate_sample.get(feature)
        original_value = original_sample.get(feature)

        freeze_violation = _freeze_violation(feature, original_sample, candidate_sample)
        if freeze_violation:
            frozen_features_changed.append(feature)
            violations.append(freeze_violation)

        if metadata["data_type"] == "categorical":
            if not _categorical_valid(feature, candidate_value):
                violations.append(f"invalid categorical value for {feature}: {candidate_value}")
        elif metadata["data_type"] == "binary":
            if candidate_value not in {0, 1, 0.0, 1.0}:
                violations.append(f"invalid binary value for {feature}: {candidate_value}")
        elif metadata["data_type"] == "integer":
            numeric_value = _coerce_numeric(candidate_value)
            if numeric_value is None:
                violations.append(f"non-numeric integer value for {feature}: {candidate_value}")
                continue
            if metadata["integer_required"] and not float(numeric_value).is_integer():
                violations.append(f"non-integer value for {feature}: {candidate_value}")
            if metadata["nonnegative_required"] and numeric_value < 0:
                violations.append(f"negative value for {feature}: {candidate_value}")
        elif metadata["data_type"] == "numeric":
            numeric_value = _coerce_numeric(candidate_value)
            if numeric_value is None:
                violations.append(f"non-numeric value for {feature}: {candidate_value}")
                continue
            if metadata["nonnegative_required"] and numeric_value < 0:
                violations.append(f"negative value for {feature}: {candidate_value}")

        if feature == "tcprtt":
            synack_val = _coerce_numeric(candidate_sample.get("synack"))
            ackdat_val = _coerce_numeric(candidate_sample.get("ackdat"))
            if synack_val is not None and ackdat_val is not None:
                expected = synack_val + ackdat_val
                actual = _coerce_numeric(candidate_value)
                if actual is not None and abs(actual - expected) > 1e-9:
                    violations.append("dependency violation: tcprtt must equal synack + ackdat")

        if feature == "is_ftp_login":
            service_val = candidate_sample.get("service")
            if service_val is not None and str(service_val) != "ftp" and candidate_value not in {0, 0.0}:
                warnings.append("FTP login flag is set while service is not ftp; check whether the session semantics remain valid")

        if feature == "ct_flw_http_mthd":
            service_val = candidate_sample.get("service")
            if service_val is not None and str(service_val) != "http" and candidate_value not in {0, 0.0}:
                warnings.append("HTTP method count is nonzero while service is not http; check service-conditioned validity")

        if feature == "is_sm_ips_ports":
            if candidate_value in {1, 1.0}:
                warnings.append("is_sm_ips_ports = 1 depends on source/destination address/port equality outside the 42-feature canonical predictor set; exact verification requires non-predictor fields")

        if feature in {"rate", "sload", "dload", "smean", "dmean"}:
            if candidate_value is not None and original_value is not None and abs(float(candidate_value) - float(original_value)) > 0:
                warnings.append(f"{feature} is a derived/summary metric and should be checked for internal consistency with neighboring flow quantities")

    domain_violations = [v for v in violations if "invalid " in v or "negative " in v or "non-numeric" in v or "non-integer" in v]
    dependency_violations = [v for v in violations if "dependency" in v]
    valid = len(violations) == 0

    return {
        "valid": valid,
        "violated_constraints": violations,
        "changed_features": changed_features,
        "frozen_features_changed": frozen_features_changed,
        "domain_violations": domain_violations,
        "dependency_violations": dependency_violations,
        "warnings": warnings,
    }


def save_constraint_table(output_path: Path) -> None:
    frame = build_feature_table_frame()
    frame.to_csv(output_path, index=False)


def _markdown_table_from_frame(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def save_constraint_specification(output_path: Path) -> None:
    frame = build_feature_table_frame()
    categories = frame["constraint_category"].value_counts().to_dict()
    markdown = f"""# UNSW-NB15 Feature Constraint Specification

## Purpose and scope

This document defines the feature-space constraints for the UNSW-NB15 explanation-spoofing research project. It does not implement any attack and does not run SHAP or LIME. It serves only as a conservative preparation layer for later input-manipulation experiments on the canonical UNSW pipeline.

The canonical predictor set consists of the 42 UNSW-NB15 features after excluding `id`, `attack_cat`, and `label`.

## Summary

- Dataset: official UNSW-NB15 training set only
- Prediction target: binary attack label
- Experimental setting: input manipulation under realistic constraints
- Attack goal: keep prediction as attack while changing the explanation substantially
- Safety principle: do not claim that every mathematically valid UNSW vector corresponds to physically realizable network traffic

## Feature category counts

- DIRECTLY_CONTROLLABLE: {categories.get('DIRECTLY_CONTROLLABLE', 0)}
- DERIVED_DEPENDENT: {categories.get('DERIVED_DEPENDENT', 0)}
- NON_CONTROLLABLE: {categories.get('NON_CONTROLLABLE', 0)}
- ATTACK_REQUIRED_PRESERVED: {categories.get('ATTACK_REQUIRED_PRESERVED', 0)}

## Core design principles

1. Feature validity is established using the canonical UNSW training data and the dataset description file, not by guessing from names alone.
2. Derived features are treated conservatively: when there is a known relationship, it is enforced or warned about; if the relation is not fully known, it is marked uncertain.
3. Contextual connection-history features are treated as dependent or frozen unless the surrounding network context is explicitly reconstructed.
4. The official test set is not used in this research-design stage.
5. The exact attacker constraints will be fixed before any attack optimization. This document is the specification layer, not the attack itself.

## Table of all 42 canonical predictors

{_markdown_table_from_frame(frame)}

## Basic vs realistic constraints

### Level 1: basic domain constraints
- no negative numeric values
- integer fields remain integers
- binary fields remain in {0, 1}
- categorical values remain among the observed valid values
- values remain inside conservative observed bounds when later used as candidate perturbation bounds

### Level 2: realistic dependency constraints
- all dependent features remain internally consistent
- protocol/service-specific conditions are respected
- contextual connection-count metrics are frozen unless recreated from an explicit traffic-history model
- attack-required properties remain preserved when the attack semantics are maintained

## Verified dependency rules

- tcprtt = synack + ackdat (explicitly stated in the feature description)
- some rate metrics and mean-size features are logically linked to bytes and packet counts, but their exact dataset-specific formula requires verification before strong enforcement
- connection-history count features are context-derived and should be treated conservatively as dependent features

## Uncertain dependencies

- exact reconstruction of rate from bytes/duration
- exact reconstruction of smean/dmean from bytes and packets
- exact network semantics of ct_state_ttl, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, and ct_src_ltm beyond their documented historical count interpretation
- exact dependence of service- and protocol-specific features on surrounding traffic context when those features are changed

## Constraints requiring external verification

- Any exact packet-level reconstruction of UNSW summary features from real network sessions
- Any exact protocol-specific dependency logic not explicitly documented in the local dataset description
- Any formal network-level feasibility rule for traffic manipulations beyond the summary-feature constraints used here

## Why realistic constraints are necessary

A mathematically valid feature vector is not automatically a physically realizable network flow. The project therefore separates what is a valid tabular perturbation from what is a realistic network event. The validator checks internal consistency and domain validity, but it does not prove packet-level realizability.

## Limitations

- UNSW-NB15 is offline benchmark data rather than live network traffic.
- Several features are engineered or count-based and may reflect dataset-generation structure.
- The current model uses a fixed official training split and does not reconstruct a time-based split because the required timestamps are not reliably available in the prepared feature set.
- Feature controllability must be justified rather than assumed.

"""
    output_path.write_text(markdown, encoding="utf-8")


def save_outputs() -> None:
    research_dir = Path(__file__).resolve().parents[2] / "results" / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    save_constraint_table(research_dir / "unsw_feature_constraints.csv")
    save_constraint_specification(research_dir / "unsw_constraint_specification.md")


if __name__ == "__main__":
    save_outputs()
