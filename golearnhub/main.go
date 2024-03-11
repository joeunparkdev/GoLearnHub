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

	// 선형 회귀 모델 생성
	model := linear_models.NewLinearRegression()

	// 모델 훈련
	err = model.Fit(rawData)
	if err != nil {
		log.Fatal("Error training model:", err)
	}

	// 2024년에 해당하는 데이터 생성
	newData := base.NewDenseInstances()

	// 새로운 데이터에 대한 예측
	prediction, err := model.Predict(newData)
	if err != nil {
		log.Fatal("Error predicting:", err)
	}

	// 결과 출력
	fmt.Printf("Prediction for 2024: %.2f\n", prediction)
}
