#!/usr/bin/zsh

EXPERIMENT=${1}

# do all thingys
python bin/process_enhanced.py enhanced/${EXPERIMENT}
python bin/convert_enhanced_to_420.py enhanced/${EXPERIMENT}
python bin/pool_work.py tasks_convert_enhanced_${EXPERIMENT}
python bin/calculate_metrics.py enhanced/${EXPERIMENT}
python bin/pool_work.py tasks_metrics_enhanced_${EXPERIMENT}

# clean
rm tasks_metrics_enhanced_${EXPERIMENT}
rm tasks_convert_enhanced_${EXPERIMENT}
rm done
rm undone