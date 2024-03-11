package main

import (
	"fmt"
	"log"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// 데이터셋 로드
	rawData, err := base.ParseCSVToInstances("data.csv", true)
	if err != nil {
		log.Fatal("Error parsing CSV:", err)
	}

	// 특성과 라벨 분리
	X, y := extractFeaturesAndLabels(rawData)

	// 선형 회귀 모델 생성
	model := linear_models.NewLinearRegression()

	// 모델 훈련
	err = model.Fit(X)
	if err != nil {
		log.Fatal("Error training model:", err)
	}

	// 2030년에 해당하는 데이터 생성
	newData := base.NewDenseInstancesFromMat64(1, []float64{2030})

	// 새로운 데이터에 대한 예측
	prediction, err := model.Predict(newData)
	if err != nil {
		log.Fatal("Error predicting:", err)
	}

	// 결과 출력
	fmt.Printf("Prediction for 2030: %.2f\n", prediction)
}

// 특성과 라벨 분리 함수
func extractFeaturesAndLabels(data base.FixedDataGrid) (base.FixedDataGrid, base.FixedDataGrid) {
	rows, _ := data.Size()

	// 특성과 라벨을 저장할 빈 데이터그리드 생성
	X := base.NewDenseInstances()
	y := base.NewDenseInstances()

	// 각 행의 데이터를 특성과 라벨로 분리하여 저장
	for i := 0; i < rows; i++ {
		instance, _ := data.GetRowVector(i)

		// 첫 번째 열은 라벨
		label := instance.GetClass()
		y.AddClass(label)
		y.AddInstance(instance)

		// 나머지 열은 특성
		instance.RemoveClass()
		X.AddInstance(instance)
	}

	return X, y
}
