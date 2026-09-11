import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.utils import to_undirected
import random

st.set_page_config(page_title="Fraud Ring Detector", layout="wide")
st.title("🛡️ Fraud Ring Detection via Localized Graph Attention Networks")
st.markdown("---")

st.sidebar.header("🔬 Simulation Parameters")
noise_level = st.sidebar.slider("Inject Structural Camouflage Noise (%)", min_value=0, max_value=50, value=10, step=5)
train_btn = st.sidebar.button("⚡ Run Stress Test Models", use_container_width=True)

@st.cache_data
def generate_base_data():
    np.random.seed(42)
    num_users = 2000
    num_features = 16
    X_np = np.random.randn(num_users, num_features).astype(np.float32)
    y_np = np.random.choice([0, 1], size=num_users, p=[0.95, 0.05])
    X_np[y_np == 1] += 0.5  
    edges_src = np.random.randint(0, num_users, size=8000)
    edges_dst = np.random.randint(0, num_users, size=8000)
    return X_np, y_np, edges_src, edges_dst

X_np, y_np, edges_src, edges_dst = generate_base_data()

def inject_camouflage(X_np, y_np, edges_src, edges_dst, pct):
    fraud_indices = np.where(y_np == 1)[0].tolist()
    benign_indices = np.where(y_np == 0)[0].tolist()
    num_alter = int(len(fraud_indices) * (pct / 100.0))
    random.seed(42)
    sampled_fraud = random.sample(fraud_indices, num_alter)
    new_src = list(edges_src)
    new_dst = list(edges_dst)
    for f in sampled_fraud:
        b = random.choice(benign_indices)
        new_src.append(f)
        new_dst.append(b)
    edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
    data = Data(x=torch.tensor(X_np), y=torch.tensor(y_np), edge_index=edge_index)
    data.edge_index = to_undirected(data.edge_index)
    return data

camou_graph = inject_camouflage(X_np, y_np, edges_src, edges_dst, noise_level)

class FraudGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

if 'trained' not in st.session_state:
    st.session_state.trained = False

if train_btn or not st.session_state.trained:
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=42, stratify=y_np)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_preds = rf.predict(X_test)
    rf_f1 = classification_report(y_test, rf_preds, output_dict=True, zero_division=0)['1']['f1-score']
    
    num_nodes = camou_graph.x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(num_nodes * 0.7)] = True
    test_mask = ~train_mask
    
    model = FraudGAT(in_channels=16, hidden_channels=8, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x_t, ei_t, y_t = camou_graph.x, camou_graph.edge_index, camou_graph.y
    
    model.train()
    for epoch in range(1, 81):
        optimizer.zero_grad()
        out = model(x_t, ei_t)
        loss = F.nll_loss(out[train_mask], y_t[train_mask].long(), weight=torch.tensor([1.0, 15.0]).float())
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        out = model(x_t, ei_t)
        gat_preds = out[test_mask].argmax(dim=1).numpy()
        gat_probs = torch.exp(out[test_mask])[:, 1].numpy()
        y_test_graph = y_t[test_mask].numpy()
        
    gat_auc = roc_auc_score(y_test_graph, gat_probs)
    gat_f1 = classification_report(y_test_graph, gat_preds, output_dict=True, zero_division=0)['1']['f1-score']
    
    st.session_state.metrics = {
        'rf_auc': rf_auc, 'rf_f1': rf_f1,
        'gat_auc': gat_auc, 'gat_f1': gat_f1,
        'original_edges': len(edges_src),
        'total_edges': camou_graph.edge_index.shape[1]
    }
    st.session_state.trained = True

metrics = st.session_state.metrics
col1, col2, col3 = st.columns(3)
col1.metric(label="📊 Graph Density (Total Edges)", value=f"{metrics['total_edges']}")
col2.metric(label="⚠️ Injected Camouflage Links", value=f"{metrics['total_edges'] - metrics['original_edges']}")
col3.metric(label="🎯 Active Attack Level", value=f"{noise_level}%")

st.markdown("### 📈 Structural Defense Performance Comparison")
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Tabular Baseline (Random Forest)")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC Score", f"{metrics['rf_auc']:.4f}")
    c2.metric("Fraud F1-Score", f"{metrics['rf_f1']:.4f}")
with col_right:
    st.subheader("Graph Attention Network (GAT)")
    c1, c2 = st.columns(2)
    c1.metric("GAT ROC-AUC Score", f"{metrics['gat_auc']:.4f}")
    c2.metric("GAT Fraud F1-Score", f"{metrics['gat_f1']:.4f}")