#!/usr/bin/env python3
"""
GPU Monitor - Monitora a utilização de GPU em tempo real
Execute enquanto o treino está rodando em outro terminal
"""

import subprocess
import time
from datetime import datetime
import sys

def get_gpu_stats():
    """Obtém estatísticas da GPU"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
        
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Erro ao obter dados de GPU: {e}")
        return None

def format_output(gpu_data):
    """Formata saída de forma legível"""
    if not gpu_data:
        print("❌ nvidia-smi não disponível")
        return False
    
    print("\n" + "="*80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏱️  {timestamp}")
    print("="*80)
    
    for line in gpu_data.split('\n'):
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) < 11:
            continue
        
        gpu_id, name, mem_used, mem_total, gpu_util, mem_util, temp, power, power_limit, gpu_clock, mem_clock = parts[:11]
        
        try:
            mem_used_f = float(mem_used)
            mem_total_f = float(mem_total)
            gpu_util_f = float(gpu_util)
            mem_util_f = float(mem_util)
            temp_f = float(temp)
            power_f = float(power)
            power_limit_f = float(power_limit)
            gpu_clock_f = float(gpu_clock)
            mem_clock_f = float(mem_clock)
            
            # Cores para visualização
            def get_color(value, thresholds=(40, 70, 90)):
                """Retorna cor baseada no valor"""
                if value >= thresholds[2]:
                    return "🔴"  # Vermelho
                elif value >= thresholds[1]:
                    return "🟠"  # Laranja
                elif value >= thresholds[0]:
                    return "🟡"  # Amarelo
                else:
                    return "🟢"  # Verde
            
            print(f"\n📊 GPU {gpu_id}: {name}")
            print("-" * 80)
            
            # Memória
            print(f"  💾 Memória:")
            print(f"     Usado: {mem_used_f:.0f}MB / {mem_total_f:.0f}MB ({mem_util_f:.1f}%) {get_color(mem_util_f)}")
            
            # Utilização
            print(f"  ⚡ Utilização:")
            print(f"     GPU: {gpu_util_f:.1f}% {get_color(gpu_util_f)}")
            print(f"     Memória: {mem_util_f:.1f}% {get_color(mem_util_f)}")
            
            # Temperatura
            temp_color = "🟢" if temp_f < 60 else "🟡" if temp_f < 75 else "🔴"
            print(f"  🌡️  Temperatura: {temp_f:.0f}°C {temp_color}")
            
            # Potência
            power_pct = (power_f / power_limit_f) * 100 if float(power_limit_f) > 0 else 0
            print(f"  🔋 Potência: {power_f:.0f}W / {power_limit_f:.0f}W ({power_pct:.0f}%) {get_color(power_pct)}")
            
            # Clock
            print(f"  ⏱️  Clock: GPU {gpu_clock_f:.0f}MHz / Mem {mem_clock_f:.0f}MHz")
            
            # Status geral
            print(f"\n  📈 Status Geral:")
            avg_util = (gpu_util_f + mem_util_f) / 2
            if avg_util > 85:
                status = "✅ Excelente"
            elif avg_util > 70:
                status = "🟡 Bom"
            elif avg_util > 50:
                status = "⚠️  Moderado"
            else:
                status = "❌ Baixo"
            
            print(f"     Utilização Média: {avg_util:.1f}% {status}")
            
        except (ValueError, IndexError) as e:
            print(f"❌ Erro ao processar dados: {e}")
            return False
    
    return True

def show_recommendations():
    """Mostra recomendações de otimização"""
    print("\n" + "="*80)
    print("💡 RECOMENDAÇÕES:")
    print("="*80)
    print("""
✅ Ideal:
   • GPU Utilization: 90-100%
   • Memory Utilization: 75-90%
   • Temperatura: 60-75°C
   • Power Draw: 85-100%

⚠️  Se abaixo do ideal:
   • GPU < 80%        → Aumentar batch_size no training_pipeline.py
   • Memória < 60%    → Aumentar batch_size
   • CPU > 50%        → Reduzir workers
   • Temperatura > 80°C → Melhorar ventilação

🚀 Para máxima potência:
   1. Maximize batch_size (até GPU atingir 100%)
   2. Use imgsz=640 (ou maior se memória permitir)
   3. Configure workers=8
   4. Habilite cache=True no treinamento
""")

def main():
    """Loop de monitoramento"""
    print("\n" + "="*80)
    print("🔍 GPU MONITOR - Cloud Architecture Detector")
    print("="*80)
    print("\n💻 Requisitos:")
    print("   • NVIDIA GPU com CUDA suportada")
    print("   • nvidia-smi instalado")
    print("   • Execute em outro terminal durante o treinamento")
    print("\n⏸️  Pressione Ctrl+C para parar\n")
    
    show_recommendations()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            gpu_data = get_gpu_stats()
            
            if gpu_data:
                format_output(gpu_data)
            else:
                print("\n❌ Erro: GPU não detectada ou nvidia-smi indisponível")
                print("   Verifique se você tem NVIDIA GPU e drivers atualizados")
                break
            
            # Clear screen a cada X iterações (terminal friendly)
            if iteration % 10 == 0:
                print("\n" + "="*80)
                print("📊 Continuando monitoramento...")
                print("="*80)
            
            time.sleep(1)  # Atualiza a cada 1 segundo
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoramento parado pelo usuário")
        print("✅ Encerrando...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
