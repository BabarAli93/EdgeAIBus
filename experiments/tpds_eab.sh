#!/bin/bash

UNIT=edgeaibus-job
SLICE=edgeaibus.slice
EXPERIMENT_CMD=  path of the python file and required arguments
DURATION=200
INTERVAL=1

RAPL_PATH="/sys/class/powercap/intel-rapl:0/energy_uj"
LOGFILE="eab_1000mc_log_$(date +%F_%T).csv"
OUT_LOG="eab_output_1000_$(date +%F_%T).log"
ERR_LOG="eab_error_1000_$(date +%F_%T).err"

echo "Time,CPU_usec_used,Memory_Bytes,Energy_uJ" > "$LOGFILE"

# Start energy and time
START_ENERGY=$(cat "$RAPL_PATH")
START_TIME=$(date +%s)

echo "Launching systemd unit: $UNIT"
systemd-run --unit="$UNIT" \
            --slice="$SLICE" \
            --property="AllowedCPUs=0" \
            --property=CPUQuota=100% \
            --property=MemoryMax=1G \
            --pty \
            bash -c "$EXPERIMENT_CMD" >"$OUT_LOG" 2>"$ERR_LOG" &

echo "Waiting for unit $UNIT to become active..."

# Wait for the systemd unit to appear and become active or inactive
while true; do
    ACTIVE_STATE=$(systemctl show -p ActiveState "$UNIT" 2>/dev/null | cut -d= -f2)
    if [[ "$ACTIVE_STATE" == "active" || "$ACTIVE_STATE" == "inactive" ]]; then
        break
    fi
    sleep 0.2
done

# Get the actual cgroup path
CGROUP_BASE=$(systemctl show -p ControlGroup "$UNIT" | cut -d'=' -f2)
CGROUP_CPU="/sys/fs/cgroup${CGROUP_BASE}/cpu.stat"
CGROUP_MEM="/sys/fs/cgroup${CGROUP_BASE}/memory.current"

echo "Monitoring resource usage..."

# Logging loop
for ((i=0; i<$DURATION; i+=$INTERVAL)); do
    TIMESTAMP=$(date +%s)

    if [ -f "$CGROUP_CPU" ] && [ -f "$CGROUP_MEM" ]; then
        CPU_USAGE=$(grep usage_usec "$CGROUP_CPU" | awk '{print $2}')
        MEM_USAGE=$(cat "$CGROUP_MEM")
    else
        CPU_USAGE="NA"
        MEM_USAGE="NA"
    fi

    CURR_ENERGY=$(cat "$RAPL_PATH")
    echo "$TIMESTAMP,$CPU_USAGE,$MEM_USAGE,$CURR_ENERGY" >> "$LOGFILE"
    sleep "$INTERVAL"

    # Optional: Exit logging early if job finished
    CURRENT_STATE=$(systemctl is-active "$UNIT")
    if [[ "$CURRENT_STATE" == "inactive" ]]; then
        echo "Systemd unit finished execution early."
        break
    fi
done

END_TIME=$(date +%s)
END_ENERGY=$(cat "$RAPL_PATH")
TOTAL_ENERGY_J=$(echo "scale=4; ($END_ENERGY - $START_ENERGY)/1000000" | bc)
TOTAL_DURATION=$((END_TIME - START_TIME))
AVG_POWER=$(echo "scale=3; $TOTAL_ENERGY_J / $TOTAL_DURATION" | bc)

echo "=== Summary ==="
echo "Duration: $TOTAL_DURATION seconds"
echo "Energy Used: $TOTAL_ENERGY_J joules"
echo "Average Power: $AVG_POWER W"
echo "Resource log saved to: $LOGFILE"
echo "Experiment stdout: $OUT_LOG"
echo "Experiment stderr: $ERR_LOG"
