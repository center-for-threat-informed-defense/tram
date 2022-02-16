/*-------------------------------------
* Functionality for uploading a report
*-------------------------------------*/

//When the Upload button is clicked, call the function to upload a report
$(document).on('change', '#id-upload',function (event){
    if(event.currentTarget) {
        let filename = event.currentTarget.files[0].name;
        if(filename){
            uploadReport();
        }
    }
});


//POST the report to the endpoint
function uploadReport() {
    let file = document.getElementById('id-upload').files[0];
    let csrf_token = document.getElementById('id-upload-form').querySelector('[name="csrfmiddlewaretoken"]').value;
    let fd = new FormData();
    fd.append('csrfmiddlewaretoken', csrf_token);
    fd.append('file', file);
    $.ajax({
         type: 'POST',
         url: '/upload/',
         data: fd,
         processData: false,
         contentType: false,
         success: function(response) {
            console.log(response)
         },
    }).done(function (){
        location.reload();
    })
}


function addMapping(attack_ids, sentence_id, report_id) {

    // If report id is passed in, use that. If not use constant provided.
    if (report_id == false) {
        report_id = REPORT_ID
    }

    attack_ids.forEach((attack_id) => {
        var data = {report: report_id, sentence: sentence_id, attack_id: attack_id, confidence: 100.0};

        $.ajax({
            type: "POST",
            url: `/api/mappings/`,
            dataType: "json",
            data: JSON.stringify(data),
            contentType:"application/json; charset=utf-8",
            headers: {
                "X-CSRFToken": CSRF_TOKEN
            },
            success: function (data) {
                // Clear select cache
                $('.select2-use').val(null).trigger('change');
                loadSentences(sentence_id);
            },
            failure: function (data) {
                console.log(`Failure: ${data}`);
            }
    })});
}

// Updates sentence and redisplays sentences, loads next sentence if applicable
function updateSentence(sentence_id, disposition, next_sentence) {
    $.ajax({
        type: "PATCH",
        url: `/api/sentences/${sentence_id}/`,
        dataType: "json",
        contentType:"application/json; charset=utf-8",
        data: JSON.stringify(disposition), // {disposition: "accept" | null}
        headers: {
            "X-CSRFToken": CSRF_TOKEN
        },
        success: function (data) {
            // If disposition is "accept", load the next sentence
            if (disposition.disposition == "accept"){
                new_sentence_id = String(parseInt(sentence_id) + 1)

                // If there is a provided next sentence, use that instead
                if (next_sentence) {
                    new_sentence_id = next_sentence;
                }
                loadSentences(new_sentence_id);
            }
            else {
                loadSentences(sentence_id);
            }
        },
        failure: function (data) {
            console.log(`Failure: ${data}`);
        }
    });
}

function deleteJob(report_id) {
    $.ajax({
        type: 'DELETE',
        url: 'api/jobs/' + report_id + '/',
        headers: {
            "X-CSRFToken": CSRF_TOKEN
        }
    }).done(function(){
        location.reload()
    })
}

function deleteMapping(sentence_id, mapping_id, refreshSentences) {
    $.ajax({
        type: "DELETE",
        url: `/api/mappings/${mapping_id}/`,
        dataType: "json",
        headers: {
            "X-CSRFToken": CSRF_TOKEN,
        },
        success: function (data) {
            // For technique_sentences, we want to refresh the page instead 
            // of reloading same sentence
            if (refreshSentences) {
                loadSentences();
            }
            else {
                loadSentences(sentence_id);
            }
        },
        failure: function (data) {
            console.log(`Failure: ${data}`);
        }
    });
}

function deleteReport(report_id) {
    $.ajax({
        type: 'DELETE',
        url: 'api/reports/' + report_id + '/',
        headers: {
            "X-CSRFToken": CSRF_TOKEN
        }
    }).done(function(){
        location.reload()
    })
}