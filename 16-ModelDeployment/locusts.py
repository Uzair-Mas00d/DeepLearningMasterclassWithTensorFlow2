from locust import SequentialTaskSet, HttpUser, task

class DetectorTask(SequentialTaskSet):
    @task
    def detection(self):
        with open("test_img.jpg", "rb") as image:
            self.client.post(
                "/detect",
                files={"im": image}
                )
            
class LoadTester(HttpUser):
    host="http://127.0.0.1:8000"
    tasks=[DetectorTask]