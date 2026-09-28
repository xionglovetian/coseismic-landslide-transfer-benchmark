param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$evaluator=Join-Path $project 'scripts\evaluate_fewshot_split.py'
$logs=Join-Path $project 'logs\e4_query_eval'
$rawRoot=Join-Path $project 'reports\e4_query_raw'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$regions=@('hokkaido_iburi_tobu','lombok','palu')
$seeds=@(42,2026,777)
$buffers=@(0,256,512)
foreach($seed in $seeds){
  foreach($region in $regions){
    foreach($buffer in $buffers){
      if($buffer -eq 0){
        $eval=if($seed -eq 42){ Join-Path $project 'data\processed\fewshot_target_splits\fewshot_eval_seed42.csv' } else { Join-Path $project 'data\processed\fewshot_target_splits\fewshot_eval_seed2026_777.csv' }
      } else {
        $eval=Join-Path $project "data\processed\fewshot_target_splits\buffered\eval_seed${seed}_${region}_buffer${buffer}.csv"
      }
      $source="D:\landslide_unet_project\outputs\bench_v2_resunet_seed${seed}\best_model.pth"
      $sourceLabel="source_seed${seed}"
      $outDir=Join-Path $rawRoot "buffer${buffer}"
      New-Item -ItemType Directory -Path $outDir -Force | Out-Null
      $out=Join-Path $outDir "${sourceLabel}_${region}.json"
      if(-not (Test-Path -LiteralPath $out)){
        $log=Join-Path $logs "${sourceLabel}_${region}_buffer${buffer}.log"
        & $python $evaluator --checkpoint $source --model ResUNet --region $region --eval-csv $eval --eval-seed $seed --checkpoint-label $sourceLabel --batch-size 16 --num-workers 4 --amp --output $out *> $log
        if($LASTEXITCODE -ne 0){ throw "E4 source evaluation failed: seed=$seed region=$region buffer=$buffer" }
        "DONE $sourceLabel $region buffer${buffer}"
      }
      foreach($mode in @('full','decoder-only')){
        $adapted="D:\landslide_unet_project\outputs\bench_v2_fewshot128_resunet_${region}_20shot_${mode}_seed${seed}\final_model.pth"
        $label="adapted_${mode}_seed${seed}"
        $out=Join-Path $outDir "${label}_${region}.json"
        if(-not (Test-Path -LiteralPath $out)){
          $log=Join-Path $logs "${label}_${region}_buffer${buffer}.log"
          & $python $evaluator --checkpoint $adapted --model ResUNet --region $region --eval-csv $eval --eval-seed $seed --checkpoint-label $label --batch-size 16 --num-workers 4 --amp --output $out *> $log
          if($LASTEXITCODE -ne 0){ throw "E4 adapted evaluation failed: seed=$seed region=$region buffer=$buffer mode=$mode" }
          "DONE $label $region buffer${buffer}"
        }
      }
    }
  }
}
'E4_QUERY_EVALUATION_COMPLETE'
