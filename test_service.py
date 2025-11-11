from services.analysis import compute_analysis_for_text

print('Running compute_analysis_for_text (short test)')
res = compute_analysis_for_text('I always mess everything up and nothing ever works')
print('Keys in result:', list(res.keys()))
print('overallScore:', res.get('overallScore'))
print('model keys:', list(res.get('model', {}).keys()))
