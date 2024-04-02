using HerringstackApi.Abstractions;
using HerringstackApi.Services;
using Microsoft.AspNetCore.Mvc;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace HerringstackApi.Controllers;

[Route("api/[controller]")]
[ApiController]
public class LatentCxrImageController : ControllerBase
{
    private readonly ICxr8DataManager _dataManager;

    public LatentCxrImageController(ICxr8DataManager dataManager)
    {
        _dataManager = dataManager;
    }

    // GET: api/<LatentCxrImageController>/subjects
    [HttpGet("subjects")]
    public async Task<IActionResult> Get([FromQuery] SubjectFilter subjectFilter)
    {
        var subjectItems = await _dataManager.GetSubjectItemsAsync(subjectFilter.PageSize, subjectFilter.PageNumber);
        return Ok(subjectItems);
    }

    // GET api/<LatentCxrImageController>/subjects/5/images
    [HttpGet("subjects/{id}/images")]
    public async Task<IActionResult> GetSubjectImages(int id)
    {
        var images = await _dataManager.GetImagesForSubjectAsync(id);
        return Ok(images);
    }

    // POST api/<LatentCxrImageController>
    [HttpPost]
    public void Post([FromBody] string value)
    {
    }

    // PUT api/<LatentCxrImageController>/5
    [HttpPut("{id}")]
    public void Put(int id, [FromBody] string value)
    {
    }
    
    // DELETE api/<LatentCxrImageController>/5
    [HttpDelete("{id}")]
    public void Delete(int id)
    {
    }
}
