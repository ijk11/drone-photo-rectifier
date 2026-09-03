# samples

테스트용 이미지를 두는 폴더입니다. 용량이 큰 원본 사진은 git에 커밋되지 않도록
`.gitignore` 에서 제외하고 있습니다.

`python -m dronerect.tools.make_synthetic` 으로 알려진 왜곡을 넣은 합성
검증 이미지를 생성할 수 있습니다(정답 파라미터가 함께 저장되므로 보정 정확도를
숫자로 검증할 수 있습니다).
