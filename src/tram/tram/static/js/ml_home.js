/* Functionality for various buttons on ml home page */

function retrainModel(model_id) {
    url = "/ml/retrain/" + model_id;
    $.ajax({
        type: "POST",
        url: url,
        headers: {
            "X-CSRFToken": CSRF_TOKEN,
        },
        success: function (response) {
            console.log(response);
            location.reload();
        },
        failure: function (data) {
            console.log(`Failure: ${data}`);
        }
    });
}


